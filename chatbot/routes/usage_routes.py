"""
Token usage and cost reporting.

Live numbers come from the in-process tracker (fast, resets on restart);
historical numbers come from the token_usage table.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from config.pricing import known_models
from config.settings import settings
from services import cost_optimizer
from services.usage_tracker import usage_tracker
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/usage", tags=["Usage & Cost"])


@router.get("/summary")
async def usage_summary():
    """Live totals for every seller since the process started."""
    sellers = usage_tracker.all_sellers()
    return {
        "note": "Live in-process totals; they reset when the app restarts. Use /usage/history for durable figures.",
        "tracking_enabled": settings.track_token_usage,
        "sellers": sellers,
        "totals": {
            "sellers": len(sellers),
            "cost_usd": round(sum(s["cost_usd"] for s in sellers.values()), 6),
            "total_tokens": sum(s["total_tokens"] for s in sellers.values()),
            "turns": sum(s["turns"] for s in sellers.values()),
        },
    }


@router.get("/seller/{seller_id}")
async def seller_usage(seller_id: str):
    """One seller's live usage, plus today's spend against their budget."""
    spent_today = usage_tracker.spend_today(seller_id)
    budget = settings.daily_budget_usd
    return {
        "seller_id": seller_id,
        "usage": usage_tracker.seller_usage(seller_id),
        "today": {
            "cost_usd": round(spent_today, 6),
            "budget_usd": budget or None,
            "budget_remaining_usd": round(max(0.0, budget - spent_today), 6) if budget else None,
            "over_budget": bool(budget and spent_today >= budget),
        },
    }


@router.get("/session/{seller_id}/{user_id}")
async def session_usage(seller_id: str, user_id: str):
    """What one customer's conversation has cost so far."""
    return {
        "seller_id": seller_id,
        "user_id": user_id,
        "usage": usage_tracker.session_usage(seller_id, user_id),
    }


@router.get("/sessions/top")
async def top_sessions(limit: int = Query(default=20, ge=1, le=200)):
    """The most expensive live conversations - useful for spotting a loop."""
    return {"sessions": usage_tracker.top_sessions(limit)}


@router.get("/models")
async def model_pricing():
    """Configured price per 1M tokens for each known model."""
    return {
        "note": "USD per 1M tokens. Override or extend via the MODEL_PRICING_FILE env var.",
        "pricing": known_models(),
    }


@router.get("/optimization")
async def optimization_status():
    """What cost controls are switched on, and how well they're working."""
    return cost_optimizer.status()


def _history_rows(seller_id: Optional[str], days: int, group_by: str) -> List[Dict[str, Any]]:
    """Aggregate token_usage over a window. Read-only, parameterised SQL."""
    from db.database import read_session

    # group_by is validated by the caller against this map, never interpolated raw.
    grouping = {
        "day": "date_trunc('day', created_at)::date",
        "model": "model",
        "customer": "customer_id",
        "seller": "seller_id",
    }[group_by]

    where = "created_at >= NOW() - (:days || ' days')::INTERVAL"
    params: Dict[str, Any] = {"days": days}
    if seller_id:
        where += " AND seller_id = :seller_id"
        params["seller_id"] = str(seller_id)

    sql = text(
        f"""
        SELECT {grouping} AS bucket,
               COUNT(*)               AS turns,
               SUM(llm_calls)         AS llm_calls,
               SUM(prompt_tokens)     AS prompt_tokens,
               SUM(completion_tokens) AS completion_tokens,
               SUM(total_tokens)      AS total_tokens,
               SUM(cost_usd)          AS cost_usd
          FROM token_usage
         WHERE {where}
         GROUP BY bucket
         ORDER BY bucket DESC
        """
    )

    with read_session() as db:
        rows = db.execute(sql, params).mappings().fetchall()

    return [
        {
            "bucket": str(row["bucket"]),
            "turns": int(row["turns"] or 0),
            "llm_calls": int(row["llm_calls"] or 0),
            "prompt_tokens": int(row["prompt_tokens"] or 0),
            "completion_tokens": int(row["completion_tokens"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "cost_usd": float(row["cost_usd"] or 0),
        }
        for row in rows
    ]


@router.get("/history")
async def usage_history(
    seller_id: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
    group_by: str = Query(default="day", pattern="^(day|model|customer|seller)$"),
):
    """Durable usage from the token_usage table, grouped as requested."""
    try:
        rows = await run_in_threadpool(_history_rows, seller_id, days, group_by)
    except Exception as e:
        logger.error("Could not read usage history: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not read usage history. Has "
                "database/migrations/004_token_usage_and_templates.sql been applied?"
            ),
        )

    return {
        "seller_id": seller_id,
        "days": days,
        "group_by": group_by,
        "rows": rows,
        "totals": {
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "turns": sum(r["turns"] for r in rows),
        },
    }
