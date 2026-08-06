"""Action parsing and validation for MegaGem."""

import json
import re
from dataclasses import dataclass
from typing import Any

from .cards import AuctionType, GemColor
from .state import GameState


PARSE_METHOD_STRICT_JSON = "strict_json"
PARSE_METHOD_CODE_BLOCK = "code_block"
PARSE_METHOD_BRACE_MATCH = "brace_match"
PARSE_METHOD_PROSE_FALLBACK = "prose_fallback"
PARSE_METHOD_NONE = "none"


@dataclass
class ParsedBid:
    """Parsed bid action from a player."""

    bid: int
    valid: bool
    parse_method: str = PARSE_METHOD_NONE
    error: str = ""
    raw_response: str = ""
    reasoning: str = ""


@dataclass
class ParsedReveal:
    """Parsed gem reveal action from a player."""

    gem_color: str
    valid: bool
    parse_method: str = PARSE_METHOD_NONE
    error: str = ""
    raw_response: str = ""
    reasoning: str = ""


@dataclass
class ParsedMissionClaim:
    """Parsed mission claim action from a player."""

    mission_ids: list[str]
    valid: bool
    parse_method: str = PARSE_METHOD_NONE
    error: str = ""
    raw_response: str = ""
    reasoning: str = ""


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from model output, keeping the content inside."""
    return re.sub(r"</?think>", "", text).strip()


def extract_reasoning_from_text(text: str) -> str:
    """Return text before the JSON object (strips <think> tags first)."""
    # Strip think tags before processing
    text = strip_think_tags(text)

    # Try direct JSON parse first to see if entire text is JSON
    try:
        json.loads(text.strip())
        # If entire text is JSON, no reasoning
        return ""
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks first (most common format)
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        json_start = match.start()
        reasoning = text[:json_start].strip()
        # Clean up common prefixes
        reasoning = re.sub(r"^(reasoning|reason|thinking|analysis):\s*", "", reasoning, flags=re.IGNORECASE)
        return reasoning

    # Find JSON by trying to parse from each position where we see '{'
    # This handles nested JSON by validating the parse
    json_start = None
    text_stripped = text.strip()

    for i in range(len(text_stripped)):
        if text_stripped[i] == "{":
            # Try to parse JSON starting from this position
            # We'll try to find the end by counting braces
            brace_count = 0
            j = i
            found_end = False

            while j < len(text_stripped):
                if text_stripped[j] == "{":
                    brace_count += 1
                elif text_stripped[j] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace, try to parse
                        try:
                            potential_json = text_stripped[i : j + 1]
                            json.loads(potential_json)  # Validate it's valid JSON
                            json_start = i
                            found_end = True
                            break
                        except json.JSONDecodeError:
                            break
                j += 1

            if found_end:
                break

    if json_start is None:
        # No valid JSON found, return entire text as reasoning
        reasoning = text.strip()
    else:
        # Extract everything before JSON
        # Account for leading whitespace stripped from text
        leading_ws = len(text) - len(text.lstrip())
        original_start = leading_ws + json_start
        reasoning = text[:original_start].strip()

    # Clean up common prefixes
    reasoning = re.sub(r"^(reasoning|reason|thinking|analysis):\s*", "", reasoning, flags=re.IGNORECASE)

    return reasoning


def extract_json_with_method(text: str) -> tuple[dict[str, Any] | None, str, tuple[int, int] | None]:
    """Extract JSON from ``text``, returning ``(data, method, span)``.

    ``span`` is into ``text`` after ``strip_think_tags``; ``None`` if no match.
    """
    text = strip_think_tags(text)

    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            start = text.find(stripped)
            if start != -1:
                return data, PARSE_METHOD_STRICT_JSON, (start, start + len(stripped))
            return data, PARSE_METHOD_STRICT_JSON, None
    except json.JSONDecodeError:
        pass

    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data, PARSE_METHOD_CODE_BLOCK, (match.start(1), match.end(1))
        except json.JSONDecodeError:
            pass

    text_stripped = text.strip()
    leading_ws = len(text) - len(text.lstrip())
    for i in range(len(text_stripped)):
        if text_stripped[i] == "{":
            brace_count = 0
            j = i
            while j < len(text_stripped):
                if text_stripped[j] == "{":
                    brace_count += 1
                elif text_stripped[j] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = text_stripped[i : j + 1]
                        try:
                            data = json.loads(potential_json)
                            if isinstance(data, dict):
                                return data, PARSE_METHOD_BRACE_MATCH, (
                                    leading_ws + i,
                                    leading_ws + j + 1,
                                )
                        except json.JSONDecodeError:
                            pass
                        break
                j += 1

    return None, PARSE_METHOD_NONE, None


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text, handling markdown code blocks, nested JSON, and <think> tags."""
    data, _method, _span = extract_json_with_method(text)
    return data


def length_split(response: str) -> dict[str, int]:
    """Return ``{total_chars, think_block_chars, json_chars}`` for ``response``.

    ``think_block_chars`` sums the contents of literal ``<think>...</think>``
    blocks (excluding tags); ``json_chars`` is the matched JSON span length.
    """
    think_block_chars = sum(
        len(m.group(1)) for m in re.finditer(r"<think>(.*?)</think>", response, re.DOTALL)
    )
    _data, _method, span = extract_json_with_method(response)
    json_chars = (span[1] - span[0]) if span is not None else 0

    return {
        "total_chars": len(response),
        "think_block_chars": think_block_chars,
        "json_chars": json_chars,
    }


def _extract_bid_from_reasoning(text: str) -> int | None:
    """Prose-fallback bid extraction; returns the rightmost match (final decision)."""
    patterns = [
        r"(?:I(?:'ll| will| should| shall|'m going to) bid)\s*\*{0,2}(\d+)\*{0,2}",
        r"(?:bid(?:ding)?)\s*\*{0,2}(\d+)\*{0,2}\s*coins?",
        r"(?:my bid(?:\s+is)?:?\s*)\*{0,2}(\d+)\*{0,2}",
        r'(?:bid)\s*(?:of|=)\s*\*{0,2}(\d+)\*{0,2}',
    ]
    # Collect all matches with their position in the text
    matches: list[tuple[int, int]] = []  # (position, bid)
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            bid = int(m.group(1))
            if 0 <= bid <= 200:  # sanity check
                matches.append((m.start(), bid))
    if not matches:
        return None
    # Return the bid from the last (rightmost) match in the text
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def parse_bid(response: str) -> ParsedBid:
    """Parse a player's bid. Expects reasoning then ``{"bid": <int>}``."""
    reasoning = extract_reasoning_from_text(response)
    data, method, _span = extract_json_with_method(response)

    if data is None:
        fallback_bid = _extract_bid_from_reasoning(response)
        if fallback_bid is not None:
            return ParsedBid(
                bid=fallback_bid,
                valid=True,
                parse_method=PARSE_METHOD_PROSE_FALLBACK,
                raw_response=response,
                reasoning=reasoning,
            )
        return ParsedBid(
            bid=0,
            valid=False,
            parse_method=PARSE_METHOD_NONE,
            error="Could not parse JSON from response",
            raw_response=response,
            reasoning=reasoning,
        )

    if "bid" not in data:
        return ParsedBid(
            bid=0,
            valid=False,
            parse_method=method,
            error="Response missing 'bid' field",
            raw_response=response,
            reasoning=reasoning,
        )

    bid_value = data["bid"]

    if not isinstance(bid_value, (int, float)):
        return ParsedBid(
            bid=0,
            valid=False,
            parse_method=method,
            error=f"Bid must be a number, got {type(bid_value).__name__}",
            raw_response=response,
            reasoning=reasoning,
        )

    # Negative bids are parseable; legality is checked by validate_bid_for_auction.
    return ParsedBid(
        bid=int(bid_value),
        valid=True,
        parse_method=method,
        raw_response=response,
        reasoning=reasoning,
    )


def parse_reveal(response: str) -> ParsedReveal:
    """Parse ``{"reveal": "<color>"}``. Hand-membership check lives in
    ``validate_reveal_for_hand``, not here."""
    reasoning = extract_reasoning_from_text(response)
    data, method, _span = extract_json_with_method(response)

    if data is None:
        return ParsedReveal(
            gem_color="",
            valid=False,
            parse_method=PARSE_METHOD_NONE,
            error="Could not parse JSON from response",
            raw_response=response,
            reasoning=reasoning,
        )

    if "reveal" not in data:
        return ParsedReveal(
            gem_color="",
            valid=False,
            parse_method=method,
            error="Response missing 'reveal' field",
            raw_response=response,
            reasoning=reasoning,
        )

    color = data["reveal"]

    if not isinstance(color, str):
        return ParsedReveal(
            gem_color="",
            valid=False,
            parse_method=method,
            error=f"Reveal must be a string, got {type(color).__name__}",
            raw_response=response,
            reasoning=reasoning,
        )

    color_normalized = color.strip().title()

    valid_colors = [c.value for c in GemColor]
    if color_normalized not in valid_colors:
        return ParsedReveal(
            gem_color="",
            valid=False,
            parse_method=method,
            error=f"Invalid gem color '{color}'. Valid colors: {valid_colors}",
            raw_response=response,
            reasoning=reasoning,
        )

    return ParsedReveal(
        gem_color=color_normalized,
        valid=True,
        parse_method=method,
        raw_response=response,
        reasoning=reasoning,
    )


def validate_reveal_for_hand(player_hand: list[str], gem_color: str) -> tuple[bool, str]:
    """Return ``(legal, error)`` for revealing ``gem_color`` from ``player_hand``."""
    if gem_color not in player_hand:
        return False, f"You don't have {gem_color} in your hand. Your hand: {player_hand}"
    return True, ""


def parse_mission_claim(response: str, game_state: GameState, player_id: int) -> ParsedMissionClaim:
    """
    Parse mission claim(s) from a player's response.
    Expected format: reasoning text followed by {"complete_missions": ["mission_id1", "mission_id2"]} or {"complete_missions": []}
    """
    reasoning = extract_reasoning_from_text(response)
    data = extract_json_from_text(response)

    if data is None:
        return ParsedMissionClaim(
            mission_ids=[],
            valid=False,
            error="Could not parse JSON from response",
            raw_response=response,
            reasoning=reasoning,
        )

    if "complete_missions" not in data:
        return ParsedMissionClaim(
            mission_ids=[],
            valid=False,
            error="Response missing 'complete_missions' field",
            raw_response=response,
            reasoning=reasoning,
        )

    missions = data["complete_missions"]

    if not isinstance(missions, list):
        return ParsedMissionClaim(
            mission_ids=[],
            valid=False,
            error=f"complete_missions must be a list, got {type(missions).__name__}",
            raw_response=response,
            reasoning=reasoning,
        )

    # Empty list is valid (player chooses not to claim)
    if not missions:
        return ParsedMissionClaim(mission_ids=[], valid=True, raw_response=response, reasoning=reasoning)

    # Validate each mission ID
    available_ids = [m.id for m in game_state.available_missions]
    validated_missions = []

    for mission_id in missions:
        if not isinstance(mission_id, str):
            continue
        if mission_id not in available_ids:
            continue
        validated_missions.append(mission_id)

    return ParsedMissionClaim(mission_ids=validated_missions, valid=True, raw_response=response, reasoning=reasoning)


def validate_bid_for_auction(game_state: GameState, player_id: int, bid: int) -> tuple[bool, str]:
    """
    Validate a bid against game rules.
    Returns (valid, error_message).
    """
    player = game_state.players[player_id]
    auction = game_state.current_auction

    if auction is None:
        return False, "No auction in progress"

    if bid < 0:
        return False, "Bid cannot be negative"

    # For loan cards, players can bid up to (current_coins + loan_amount)
    # Example: 7 coins + 10 loan = can bid up to 17
    if auction.type == AuctionType.LOAN:
        max_bid = player.coins + auction.amount
        if bid > max_bid:
            return (
                False,
                f"Cannot bid {bid} coins. Maximum bid for this loan: {max_bid} "
                f"(you have {player.coins} + loan amount {auction.amount})",
            )
        return True, ""

    # For other cards, cannot bid more than current coins
    if bid > player.coins:
        return (
            False,
            f"Cannot bid {bid} coins, you only have {player.coins}. Only Loan cards allow bidding more than you have.",
        )

    return True, ""


def get_default_bid() -> int:
    """Return default bid when parsing fails."""
    return 0


def get_default_reveal(player_hand: list[str]) -> str:
    """Default reveal when parsing fails — deterministic (canonical first color)
    so play is reproducible, matching get_default_bid()."""
    return sorted(player_hand)[0] if player_hand else ""
