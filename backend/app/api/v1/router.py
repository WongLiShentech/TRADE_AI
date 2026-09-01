from fastapi import APIRouter

from app.api.v1 import (
    backtester,
    bot,
    candles,
    indicators,
    instruments,
    journal,
    orders,
    portfolio,
    prices,
    shadow,
    signals,
    trades,
    wiki,
)

router = APIRouter()

router.include_router(instruments.router, prefix="/instruments", tags=["instruments"])
router.include_router(candles.router, prefix="/candles", tags=["candles"])
router.include_router(prices.router, prefix="/prices", tags=["prices"])
router.include_router(indicators.router, prefix="/indicators", tags=["indicators"])
router.include_router(signals.router, prefix="/signals", tags=["signals"])
router.include_router(orders.router, prefix="/orders", tags=["orders"])
router.include_router(trades.router, prefix="/trades", tags=["trades"])
router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
router.include_router(backtester.router, prefix="/backtester", tags=["backtester"])
router.include_router(journal.router, prefix="/journal", tags=["journal"])
router.include_router(wiki.router, prefix="/wiki", tags=["wiki"])
router.include_router(bot.router, prefix="/bot", tags=["bot"])
router.include_router(shadow.router, prefix="/shadow", tags=["shadow"])
