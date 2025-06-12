#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example showing how to rank recalled items with ``ItemRanker``."""

import asyncio
import os

from metagpt.logs import logger
from metagpt.recommender import ItemRanker


async def main() -> None:
    user_profile = "性别：女，年龄：28，兴趣：时尚、美妆、旅行"
    context = "618年中大促正在进行，用户浏览了几款化妆品和旅行背包"
    items = [
        {"title": "雅诗兰黛粉底液", "desc": "遮瑕持久，自然服帖"},
        {"title": "小米行李箱", "desc": "20寸登机箱，轻盈耐用"},
        {"title": "优衣库连衣裙", "desc": "夏日清爽风格，舒适百搭"},
        {"title": "戴森吹风机", "desc": "快速干发，护发不伤发"},
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    ranker = ItemRanker(api_key=api_key)
    ranked_items = await ranker.rank(user_profile, context, items)
    for item in ranked_items:
        logger.info(item)


if __name__ == "__main__":
    asyncio.run(main())
