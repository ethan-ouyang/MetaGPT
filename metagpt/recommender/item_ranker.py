"""Tools for ranking recalled items with GPT models."""

import json
from importlib import import_module
from typing import Any, Dict, List


class ItemRanker:
    """Rank items using GPT-4o-mini or a specified LLM model."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        LLM = import_module("metagpt.llm").LLM
        LLMConfig = import_module("metagpt.configs.llm_config").LLMConfig
        self.llm = LLM(LLMConfig(api_key=api_key, model=model))

    async def rank(
        self, user_profile: str, context: str, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return ``items`` sorted from high to low relevance."""
        if not items:
            return []

        item_desc = "\n".join(
            f"{idx + 1}. {item.get('title', '')}: {item.get('desc', '')}"
            for idx, item in enumerate(items)
        )
        prompt = (
            "You are an expert recommendation ranker. Given the user profile, current context "
            "and candidate items, rank the items in descending order of relevance to the user. "
            "Return only a JSON list of item numbers.\n\n"
            f"## User Profile\n{user_profile}\n\n"
            f"## Context\n{context}\n\n"
            f"## Items\n{item_desc}\n"
        )

        rsp = await self.llm.aask(prompt, stream=False)

        try:
            ranked_indexes = [int(x) for x in json.loads(rsp)]
        except Exception:
            ranked_indexes = []

        ranked_items = [items[i - 1] for i in ranked_indexes if 1 <= i <= len(items)]
        remaining = [item for idx, item in enumerate(items, start=1) if idx not in ranked_indexes]
        return ranked_items + remaining


__all__ = ["ItemRanker"]

