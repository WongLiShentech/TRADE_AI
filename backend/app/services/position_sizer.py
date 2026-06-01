import math


class PositionSizer:
    """Locked triangle: risk_pct + stop_distance → units. Units is always the output, never the input."""

    def calculate(
        self,
        risk_amount: float,
        stop_distance_pips: float,
        pip_value_per_unit: float,
    ) -> int:
        """Returns floor(risk_amount / (stop_distance_pips * pip_value_per_unit))."""
        return math.floor(risk_amount / (stop_distance_pips * pip_value_per_unit))
