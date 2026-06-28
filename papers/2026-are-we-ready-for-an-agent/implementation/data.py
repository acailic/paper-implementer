"""
data.py — Toy LoCoMo-style multi-session dataset + dataloader.

A miniature long-conversation, multi-session QA benchmark in the spirit of
LoCoMo / LongMemEval: 4 sessions of user<->assistant dialogue, with facts that
get REVISED across sessions (e.g. "I live in Paris" -> later "I moved to
London"), and ~10 QA queries (temporal reasoning + multi-hop entity) each with
gold evidence + gold answer for substring-EM scoring.

Standalone, deterministic, no external files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# A dialogue turn                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    text: str
    session_id: int
    timestamp: float   # synthetic monotonic clock


@dataclass
class Session:
    session_id: int
    turns: List[Turn] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# A QA item with gold evidence (which session/turn) + gold answer             #
# --------------------------------------------------------------------------- #


@dataclass
class QAItem:
    query: str
    gold_answer: str                 # substring-EM target
    gold_session_ids: List[int]      # sessions that hold the evidence
    gold_keywords: List[str]         # terms that MUST appear in retrieved text
    note: str = ""                   # what it tests


# --------------------------------------------------------------------------- #
# The toy benchmark                                                           #
# --------------------------------------------------------------------------- #


def load_sessions() -> List[Session]:
    """Build 4 sessions. Timestamps are a synthetic monotonic clock (days).

    Narrative: a user (Alice) talks to an assistant over 4 sessions. Facts
    EVOLVE: she lives in Paris, then moves to London; she works as a nurse,
    then becomes a doctor; she has a dog Rex, then gets a cat Luna. This
    exercises Module U (multi-versioning / invalidation) and temporal queries.
    """
    t = 0.0
    sessions: List[Session] = []

    def add(sid: int, pairs: List[Tuple[str, str]]) -> None:
        nonlocal t
        s = Session(session_id=sid)
        for role, text in pairs:
            t += 1.0
            s.turns.append(Turn(role=role, text=text, session_id=sid, timestamp=t))
        sessions.append(s)

    # ---- Session 0 : initial facts -------------------------------------- #
    add(
        0,
        [
            ("user", "Hi, I'm Alice. I live in Paris."),
            ("assistant", "Hello Alice! Paris is a wonderful city."),
            ("user", "I work as a nurse at Saint-Antoine Hospital."),
            ("assistant", "That's a noble profession. Nursing is demanding but rewarding."),
            ("user", "I have a dog named Rex who loves long walks."),
            ("assistant", "Rex sounds lovely. What breed is he?"),
            ("user", "Rex is a golden retriever, very friendly."),
            ("assistant", "Golden retrievers are great companions."),
        ],
    )

    # ---- Session 1 : a new hobby + a friend ---------------------------- #
    add(
        1,
        [
            ("user", "I recently started learning the piano."),
            ("assistant", "Wonderful! How are your piano lessons going?"),
            ("user", "Going well. My teacher is a friend called Bob."),
            ("assistant", "It's nice to learn from a friend."),
            ("user", "Bob lives in Lyon and teaches on weekends."),
            ("assistant", "Weekend lessons in Lyon sound convenient."),
            ("user", "I still live in Paris, near the Louvre."),
            ("assistant", "Living near the Louvre is fantastic."),
        ],
    )

    # ---- Session 2 : REVISIONS — location + job ------------------------ #
    add(
        2,
        [
            ("user", "Big news: I moved to London last month."),
            ("assistant", "Congratulations on the move! How is London?"),
            ("user", "I now work as a doctor at King's College Hospital."),
            ("assistant", "Quite a promotion from nursing to doctor!"),
            ("user", "I started medical school two years ago."),
            ("assistant", "That explains the career change."),
            ("user", "Rex is happy in London, he loves Hyde Park."),
            ("assistant", "Hyde Park is perfect for a golden retriever."),
        ],
    )

    # ---- Session 3 : a new pet + another revision ---------------------- #
    add(
        3,
        [
            ("user", "I adopted a cat named Luna yesterday."),
            ("assistant", "Congratulations on adopting Luna!"),
            ("user", "Luna is a british shorthair, very calm."),
            ("assistant", "British shorthairs are known for being calm."),
            ("user", "Rex and Luna get along surprisingly well."),
            ("assistant", "That's great that they're friends."),
            ("user", "I like cooking italian food on weekends now."),
            ("assistant", "Italian cooking is delicious. What's your favorite dish?"),
        ],
    )

    return sessions


def load_queries() -> List[QAItem]:
    """10 QA items: temporal reasoning (current vs stale facts) + multi-hop
    entity (Bob -> Lyon, Rex -> breed/park). Gold answers reflect the LATEST
    (valid) fact so we can show invalidation works."""
    return [
        # --- temporal: revised facts (must hit the NEW value, not the old) ---
        QAItem(
            query="Where does Alice live now?",
            gold_answer="London",
            gold_session_ids=[2],
            gold_keywords=["London"],
            note="temporal revision: Paris->London (must not return Paris)",
        ),
        QAItem(
            query="What is Alice's current job?",
            gold_answer="doctor",
            gold_session_ids=[2],
            gold_keywords=["doctor"],
            note="temporal revision: nurse->doctor (must not return nurse)",
        ),
        QAItem(
            query="Where does Alice work now?",
            gold_answer="King's College Hospital",
            gold_session_ids=[2],
            gold_keywords=["King", "College"],
            note="temporal revision: workplace changed",
        ),
        # --- multi-hop entity ---
        QAItem(
            query="What breed is Alice's dog Rex?",
            gold_answer="golden retriever",
            gold_session_ids=[0],
            gold_keywords=["golden retriever"],
            note="multi-hop entity: Rex -> breed",
        ),
        QAItem(
            query="Who is Bob and where does he live?",
            gold_answer="Lyon",
            gold_session_ids=[1],
            gold_keywords=["Bob", "Lyon"],
            note="multi-hop entity: Bob -> city",
        ),
        QAItem(
            query="What instrument is Alice learning?",
            gold_answer="piano",
            gold_session_ids=[1],
            gold_keywords=["piano"],
            note="single-hop factual",
        ),
        # --- new pet ---
        QAItem(
            query="What is the name of Alice's cat?",
            gold_answer="Luna",
            gold_session_ids=[3],
            gold_keywords=["Luna"],
            note="single-hop factual",
        ),
        QAItem(
            query="What breed is Alice's cat?",
            gold_answer="british shorthair",
            gold_session_ids=[3],
            gold_keywords=["british shorthair"],
            note="single-hop factual",
        ),
        # --- preference ---
        QAItem(
            query="What kind of food does Alice like to cook?",
            gold_answer="italian",
            gold_session_ids=[3],
            gold_keywords=["italian"],
            note="single-hop preference",
        ),
        # --- temporal: the OLD value, which should now be INVALID ---
        QAItem(
            query="Did Alice ever live in Paris?",
            gold_answer="Paris",
            gold_session_ids=[0, 1],
            gold_keywords=["Paris"],
            note="historical fact (old version, logically invalidated but kept)",
        ),
    ]


def dataloader() -> Tuple[List[Session], List[QAItem]]:
    return load_sessions(), load_queries()


__all__ = ["Turn", "Session", "QAItem", "load_sessions", "load_queries", "dataloader"]
