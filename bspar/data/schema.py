"""Core data structures for BSPAR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Span:
    """A text span or NULL indicator.

    For explicit spans: start/end are token indices (inclusive).
    For implicit (NULL): start=end=-1, is_null=True.
    """
    start: int
    end: int
    text: str
    is_null: bool = False

    @property
    def length(self) -> int:
        if self.is_null:
            return 0
        return self.end - self.start + 1

    @staticmethod
    def null(role: str = "") -> Span:
        return Span(start=-1, end=-1, text="NULL", is_null=True)

    def __eq__(self, other):
        if not isinstance(other, Span):
            return False
        if self.is_null and other.is_null:
            return True
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        if self.is_null:
            return hash((-1, -1, True))
        return hash((self.start, self.end, False))


@dataclass
class Quad:
    """A single aspect sentiment quadruple."""
    aspect: Span
    opinion: Span
    category: str
    sentiment: Optional[str] = None       # POS/NEG/NEU for ASQP
    valence: Optional[float] = None       # for dimABSA
    arousal: Optional[float] = None       # for dimABSA

    def matches(self, other: Quad, match_affective: bool = True) -> bool:
        """Exact match check for evaluation."""
        if self.aspect != other.aspect:
            return False
        if self.opinion != other.opinion:
            return False
        if self.category != other.category:
            return False
        if match_affective:
            if self.sentiment is not None and self.sentiment != other.sentiment:
                return False
        return True


@dataclass
class Example:
    """A single training/inference example."""
    id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_offsets: list[tuple[int, int]] = field(default_factory=list)
    quads: list[Quad] = field(default_factory=list)


@dataclass
class SpanCandidate:
    """A scored span candidate from Module A."""
    span: Span
    repr_vec: object = None     # tensor, filled at runtime
    asp_score: float = 0.0
    opn_score: float = 0.0
    role: str = ""              # "aspect" | "opinion" | "null_aspect" | "null_opinion"


@dataclass
class PairCandidate:
    """A scored pair candidate from Module B."""
    aspect: SpanCandidate
    opinion: SpanCandidate
    pair_repr: object = None    # tensor
    pair_score: float = 0.0
    category_probs: object = None
    category_pred: str = ""
    affective_pred: object = None


@dataclass
class QuadCandidate:
    """A fully expanded quad candidate for reranking."""
    pair: PairCandidate
    category: str = ""
    affective: object = None
    quad_score: float = 0.0
    # Meta features for reranker input
    asp_unary: float = 0.0
    opn_unary: float = 0.0
    pair_validity: float = 0.0
    cat_prob: float = 0.0
    cat_entropy: float = 0.0
    has_null_aspect: bool = False
    has_null_opinion: bool = False
    asp_length: int = 0
    opn_length: int = 0
    label: int = -1             # 1=gold match, 0=negative, -1=unlabeled

    def to_quad(self) -> Quad:
        """Convert to final Quad for output."""
        asp = self.pair.aspect.span
        opn = self.pair.opinion.span
        if isinstance(self.affective, str):
            return Quad(aspect=asp, opinion=opn, category=self.category,
                        sentiment=self.affective)
        elif isinstance(self.affective, (tuple, list)) and len(self.affective) == 2:
            v, ar = self.affective
            return Quad(aspect=asp, opinion=opn, category=self.category,
                        valence=float(v), arousal=float(ar))
        return Quad(aspect=asp, opinion=opn, category=self.category)

    def meta_features(self) -> list[float]:
        """Return meta feature vector for reranker."""
        return [
            self.asp_unary,
            self.opn_unary,
            self.pair_validity,
            self.cat_prob,
            self.cat_entropy,
            float(self.has_null_aspect),
            float(self.has_null_opinion),
            float(self.asp_length),
            float(self.opn_length),
        ]
