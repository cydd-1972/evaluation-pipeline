from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

_MONTHS: Final[dict[str, int]] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s*,?\s*(\d{4})\b")
_MDY_RE = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b")

_RELATIVE_SIMPLE_RE = re.compile(r"\b(yesterday|today|tomorrow)\b", re.IGNORECASE)
_RELATIVE_DAYS_AGO_RE = re.compile(
    r"\b((?:\d+)|one|two|three|four|five|six|seven)\s+days?\s+ago\b",
    re.IGNORECASE,
)
_RELATIVE_WEEKDAY_RE = re.compile(
    r"\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_RELATIVE_MONTH_RE = re.compile(r"\b(last|next)\s+month\b", re.IGNORECASE)
_THIS_MONTH_RE = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_LAST_WEEK_RE = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_WEEKEND_AGO_RE = re.compile(
    r"\b(?:last\s+weekend|((?:\d+)|one|two|three|four|five|six|seven)\s+weekends?\s+ago)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}


def parse_anchor_date(anchor_time: str) -> date | None:
    """Extract a calendar date from a session_time / anchor_time string."""
    raw = (anchor_time or "").strip()
    if not raw:
        return None

    match = _ISO_DATE_RE.search(raw)
    if match:
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return date(year, month, day)

    match = _DMY_RE.search(raw)
    if match:
        day = int(match.group(1))
        month_name = match.group(2).strip().lower()
        year = int(match.group(3))
        month = _MONTHS.get(month_name)
        if month:
            return date(year, month, day)

    match = _MDY_RE.search(raw)
    if match:
        month_name = match.group(1).strip().lower()
        day = int(match.group(2))
        year = int(match.group(3))
        month = _MONTHS.get(month_name)
        if month:
            return date(year, month, day)

    return None


@dataclass(frozen=True)
class ResolvedTime:
    value: str
    kind: str  # "date" | "month" | "date_range"


def _range_value(start: date, end: date) -> str:
    return f"{start.isoformat()} to {end.isoformat()}"


def _coerce_small_number(match: re.Match[str]) -> int | None:
    for group in match.groups():
        if not group:
            continue
        value = group.lower()
        if value.isdigit():
            return int(value)
        if value in _NUMBER_WORDS:
            return _NUMBER_WORDS[value]
    return None


def resolve_relative_time(text: str, anchor: date) -> ResolvedTime | None:
    """Resolve a small set of English relative time expressions to an absolute value.

    Returns:
      - kind="date": YYYY-MM-DD
      - kind="month": YYYY-MM
      - kind="date_range": YYYY-MM-DD to YYYY-MM-DD
    """
    if not text or anchor is None:
        return None
    raw = text.strip()
    if not raw:
        return None

    match = _RELATIVE_SIMPLE_RE.search(raw)
    if match:
        token = match.group(1).lower()
        if token == "today":
            resolved = anchor
        elif token == "yesterday":
            resolved = anchor - timedelta(days=1)
        else:
            resolved = anchor + timedelta(days=1)
        return ResolvedTime(value=resolved.isoformat(), kind="date")

    match = _RELATIVE_DAYS_AGO_RE.search(raw)
    if match:
        days = _coerce_small_number(match)
        if days is not None:
            resolved = anchor - timedelta(days=days)
            return ResolvedTime(value=resolved.isoformat(), kind="date")

    match = _LAST_WEEK_RE.search(raw)
    if match:
        current_week_start = anchor - timedelta(days=anchor.weekday())
        start = current_week_start - timedelta(days=7)
        end = start + timedelta(days=6)
        return ResolvedTime(value=_range_value(start, end), kind="date_range")

    match = _WEEKEND_AGO_RE.search(raw)
    if match:
        weekend_count = _coerce_small_number(match) or 1
        current_week_start = anchor - timedelta(days=anchor.weekday())
        saturday = current_week_start - timedelta(days=2 + 7 * (weekend_count - 1))
        sunday = saturday + timedelta(days=1)
        return ResolvedTime(value=_range_value(saturday, sunday), kind="date_range")

    match = _RELATIVE_WEEKDAY_RE.search(raw)
    if match:
        direction = match.group(1).lower()
        weekday_name = match.group(2).lower()
        target = _WEEKDAYS.get(weekday_name)
        if target is None:
            return None
        current = anchor.weekday()
        if direction == "last":
            delta = (current - target) % 7
            if delta == 0:
                delta = 7
            resolved = anchor - timedelta(days=delta)
        else:
            delta = (target - current) % 7
            if delta == 0:
                delta = 7
            resolved = anchor + timedelta(days=delta)
        return ResolvedTime(value=resolved.isoformat(), kind="date")

    match = _RELATIVE_MONTH_RE.search(raw)
    if match:
        direction = match.group(1).lower()
        year, month = anchor.year, anchor.month
        if direction == "next":
            month += 1
            if month > 12:
                month = 1
                year += 1
        else:
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        return ResolvedTime(value=f"{year:04d}-{month:02d}", kind="month")

    match = _THIS_MONTH_RE.search(raw)
    if match:
        return ResolvedTime(value=f"{anchor.year:04d}-{anchor.month:02d}", kind="month")

    return None
