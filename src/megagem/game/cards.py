"""Card dataclasses for MegaGem."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class GemColor(str, Enum):
    """Gem color types."""

    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    PURPLE = "Purple"
    YELLOW = "Yellow"


class AuctionType(str, Enum):
    """Auction card types."""

    TREASURE = "treasure"
    LOAN = "loan"
    INVESTMENT = "investment"


@dataclass
class GemCard:
    """A gem card with a color."""

    id: str
    color: GemColor

    @classmethod
    def from_dict(cls, data: dict) -> "GemCard":
        return cls(id=data["id"], color=GemColor(data["color"]))

    def to_dict(self) -> dict:
        return {"id": self.id, "color": self.color.value}


@dataclass
class AuctionCard:
    """An auction card (treasure, loan, or investment)."""

    id: str
    type: AuctionType
    gems: int = 0  # For treasure cards: 1 or 2
    amount: int = 0  # For loan cards: 10 or 20
    bonus: int = 0  # For investment cards: 5 or 10

    @classmethod
    def from_dict(cls, data: dict) -> "AuctionCard":
        return cls(
            id=data["id"],
            type=AuctionType(data["type"]),
            gems=data.get("gems", 0),
            amount=data.get("amount", 0),
            bonus=data.get("bonus", 0),
        )

    def to_dict(self) -> dict:
        result = {"id": self.id, "type": self.type.value}
        if self.type == AuctionType.TREASURE:
            result["gems"] = self.gems
        elif self.type == AuctionType.LOAN:
            result["amount"] = self.amount
        elif self.type == AuctionType.INVESTMENT:
            result["bonus"] = self.bonus
        return result

    def get_description(self) -> str:
        """Get human-readable description of the auction."""
        if self.type == AuctionType.TREASURE:
            return f"Treasure: Win {self.gems} gem(s)"
        if self.type == AuctionType.LOAN:
            return f"Loan: Receive {self.amount} coins now, pay back at end"
        return f"Investment: Pay bid now, get bid + {self.bonus} coins at end"


@dataclass
class MissionRequirement:
    """Mission requirement specification."""

    type: Literal["any_different", "any_same", "pairs", "specific", "same_color"]
    count: int = 0
    colors: list[str] = field(default_factory=list)
    color: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "MissionRequirement":
        req_type = data.get("type", "specific")
        if "colors" in data:
            req_type = "specific"
        elif "color" in data:
            req_type = "same_color"
        return cls(
            type=req_type,
            count=data.get("count", 0),
            colors=data.get("colors", []),
            color=data.get("color", ""),
        )


@dataclass
class Mission:
    """A mission card with requirements and reward."""

    id: str
    type: str
    description: str
    requirement: MissionRequirement
    reward: int

    @classmethod
    def from_dict(cls, data: dict) -> "Mission":
        return cls(
            id=data["id"],
            type=data["type"],
            description=data["description"],
            requirement=MissionRequirement.from_dict(data["requirement"]),
            reward=data["reward"],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "reward": self.reward,
        }

    def check_completion(self, collection: list[str]) -> bool:
        """Check if a collection of gem colors satisfies this mission."""
        from collections import Counter

        color_counts = Counter(collection)

        req = self.requirement

        if req.type == "any_different":
            # Need at least 'count' different colors
            return len(color_counts) >= req.count

        elif req.type == "any_same":
            # Need at least 'count' of any single color
            return any(c >= req.count for c in color_counts.values())

        elif req.type == "pairs":
            # Need 'count' pairs of DIFFERENT colors (each pair is 2+ of same color)
            # Example: 2 Red + 2 Blue = 2 pairs of different colors (valid)
            # Example: 4 Red = 0 pairs of different colors (invalid, all same color)
            # Example: 2 Red + 2 Blue + 2 Green = 3 pairs, but need only 2 (valid)
            # Count how many different colors have 2+ gems
            pairs_of_different_colors = sum(1 for c in color_counts.values() if c >= 2)
            return pairs_of_different_colors >= req.count

        elif req.type == "specific" or req.colors:
            # Need at least 1 of each specified color
            return all(color_counts.get(color, 0) >= 1 for color in req.colors)

        elif req.type == "same_color" or req.color:
            # Need 'count' of a specific color
            return color_counts.get(req.color, 0) >= req.count

        return False


@dataclass
class ValueChart:
    """A value chart mapping gem counts in display to values."""

    id: str
    description: str
    values: dict[int, int]  # count -> value_per_gem

    @classmethod
    def from_dict(cls, chart_id: str, data: dict) -> "ValueChart":
        values = {v["count"]: v["value_per_gem"] for v in data["values"]}
        return cls(id=chart_id, description=data["description"], values=values)

    def get_gem_value(self, count_in_display: int) -> int:
        """Get the value per gem for a given count in the value display."""
        # Clamp to valid range
        count = min(count_in_display, max(self.values.keys()))
        return self.values.get(count, 0)
