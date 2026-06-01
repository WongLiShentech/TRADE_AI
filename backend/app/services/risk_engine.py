from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.brokers.base import BrokerClient
from app.config import Settings
from app.models.indicator import Indicator
from app.models.instrument import Instrument
from app.services.position_sizer import PositionSizer
from app.services.signal_engine.base import SignalOutput

logger = logging.getLogger(__name__)

_CORRELATED_GROUPS: list[frozenset[str]] = [
    frozenset({"EUR_USD", "GBP_USD"}),
    frozenset({"EUR_JPY", "GBP_JPY"}),
    frozenset({"AUD_USD", "NZD_USD"}),
]


class RiskValidationError(Exception):
    pass


@dataclass
class ValidatedSignal:
    signal: SignalOutput
    units: int
    risk_amount: float
    pip_value: float


class RiskEngine:
    """Enforcement layer. Every signal must pass validate() before order placement or backtest entry."""

    def __init__(self, settings: Settings, broker: BrokerClient) -> None:
        self._settings = settings
        self._broker = broker
        self._sizer = PositionSizer()

    def validate(
        self,
        signal: SignalOutput,
        balance: float,
        db: Session,
        inst: Instrument,
    ) -> ValidatedSignal:
        """Validates signal against all risk rules. Raises RiskValidationError on any violation."""
        atr_row = (
            db.query(Indicator)
            .filter_by(instrument_id=inst.id, granularity=signal.granularity)
            .order_by(Indicator.timestamp.desc())
            .first()
        )
        if atr_row is None or atr_row.atr14 is None:
            raise RiskValidationError("NO_ATR_DATA")

        self._check_stop_distance(signal, atr_row.atr14)
        self._check_rr_ratio(signal)

        pip_value = self._broker.get_pip_value(signal.instrument, inst.pip_size)
        risk_amount = balance * self._settings.RISK_PCT_PER_TRADE
        stop_distance_pips = abs(signal.entry - signal.stop) / inst.pip_size
        units = self._sizer.calculate(risk_amount, stop_distance_pips, pip_value)

        self._check_risk_pct(risk_amount, balance)

        if units < self._settings.OANDA_MIN_UNITS:
            raise RiskValidationError("BELOW_MIN_ORDER_SIZE")

        self._check_correlation(signal, db)

        return ValidatedSignal(
            signal=signal,
            units=units,
            risk_amount=risk_amount,
            pip_value=pip_value,
        )

    def _check_rr_ratio(self, signal: SignalOutput) -> None:
        reward = abs(signal.target - signal.entry)
        risk = abs(signal.entry - signal.stop)
        if risk == 0:
            raise RiskValidationError("ZERO_STOP_DISTANCE")
        if reward / risk < self._settings.MIN_RR_RATIO:
            raise RiskValidationError("INSUFFICIENT_RR")

    def _check_stop_distance(self, signal: SignalOutput, atr: float) -> None:
        stop_distance = abs(signal.entry - signal.stop)
        if stop_distance > self._settings.ATR_MULTIPLIER_MAX * atr:
            raise RiskValidationError("STOP_TOO_WIDE")

    def _check_risk_pct(self, risk_amount: float, balance: float) -> None:
        if balance == 0:
            raise RiskValidationError("ZERO_BALANCE")
        if risk_amount / balance > self._settings.MAX_RISK_PCT_PER_TRADE:
            raise RiskValidationError("RISK_EXCEEDS_MAX")

    def _check_correlation(self, signal: SignalOutput, db: Session) -> None:
        from app.models.signal import Signal as SignalModel
        for group in _CORRELATED_GROUPS:
            if signal.instrument not in group:
                continue
            for other in group:
                if other == signal.instrument:
                    continue
                other_inst = db.query(Instrument).filter_by(symbol=other).first()
                if other_inst is None:
                    continue
                conflict = (
                    db.query(SignalModel)
                    .filter(
                        SignalModel.instrument_id == other_inst.id,
                        SignalModel.direction == signal.direction,
                        SignalModel.status.in_(["PENDING", "APPROVED", "EXECUTED"]),
                    )
                    .first()
                )
                if conflict is not None:
                    raise RiskValidationError("CORRELATED_POSITION_EXISTS")
