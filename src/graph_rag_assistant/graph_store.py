from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import networkx as nx


@dataclass(frozen=True)
class GraphRelation:
    source: str
    relation: str
    target: str
    evidence: str


class GraphStore:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_default_graph()

    def _load_default_graph(self):
        facts: List[GraphRelation] = [
            GraphRelation("Apollo 11", "landed_at", "Sea of Tranquility", "Apollo 11 landed in the Sea of Tranquility."),
            GraphRelation("Neil Armstrong", "walked_on", "Moon", "Neil Armstrong became the first person to walk on the Moon."),
            GraphRelation("Buzz Aldrin", "walked_on", "Moon", "Buzz Aldrin walked on the Moon with Neil Armstrong."),
            GraphRelation("Apollo 11", "crew", "Neil Armstrong", "Apollo 11 crew included Neil Armstrong."),
            GraphRelation("Apollo 11", "crew", "Buzz Aldrin", "Apollo 11 crew included Buzz Aldrin."),
            GraphRelation("Apollo 11", "crew", "Michael Collins", "Apollo 11 crew included Michael Collins."),
            GraphRelation("Michael Collins", "orbited", "Moon", "Michael Collins remained in lunar orbit."),
            GraphRelation("Apollo 12", "landed_at", "Ocean of Storms", "Apollo 12 landed in the Ocean of Storms."),
            GraphRelation("Charles Conrad Jr.", "landed_on", "Moon", "Charles Conrad Jr. landed on the Moon."),
            GraphRelation("Alan Bean", "landed_on", "Moon", "Alan Bean landed on the Moon."),
            GraphRelation("Apollo 13", "planned_landing_site", "Fra Mauro", "Apollo 13 was intended to land in the Fra Mauro region."),
            GraphRelation("Apollo 13", "did_not_land_at", "Moon", "Apollo 13 did not land on the Moon due to the oxygen tank explosion."),
            GraphRelation("Apollo 14", "landed_at", "Fra Mauro", "Apollo 14 landed in the Fra Mauro region."),
            GraphRelation("Apollo 15", "landed_at", "Hadley-Apennine", "Apollo 15 landed near the Hadley-Apennine region."),
            GraphRelation("Apollo 16", "landed_at", "Descartes Highlands", "Apollo 16 landed in the Descartes Highlands."),
            GraphRelation("Apollo 17", "landed_at", "Taurus-Littrow", "Apollo 17 landed at Taurus-Littrow."),
            GraphRelation("Harrison Schmitt", "landed_on", "Moon", "Harrison Schmitt landed on the Moon."),
            GraphRelation("NASA", "managed", "Apollo program", "NASA managed the Apollo program."),
            GraphRelation("Apollo 11", "part_of", "Apollo program", "Apollo 11 was part of the Apollo program."),
            GraphRelation("Apollo 12", "part_of", "Apollo program", "Apollo 12 was part of the Apollo program."),
        ]

        for fact in facts:
            self.graph.add_edge(fact.source, fact.target, relation=fact.relation, evidence=fact.evidence)

    def query(self, question: str) -> List[dict]:
        q = question.lower()
        results = []
        for source, target, data in self.graph.edges(data=True):
            if (source.lower() in q or target.lower() in q) and ("apollo" in q or "moon" in q or "neil" in q or "fra" in q or "harrison" in q):
                results.append({
                    "source": source,
                    "relation": data["relation"],
                    "target": target,
                    "evidence": data["evidence"],
                })
        return results

    def query_related(self, entity: str, max_depth: int = 2) -> List[Tuple[str, str, str]]:
        if entity not in self.graph:
            return []
        results = []
        for source, target, data in self.graph.edges(data=True):
            if source == entity:
                results.append((source, data["relation"], target))
        return results

    def path_between(self, start: str, end: str) -> List[str | None]:
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return []

    def entities(self) -> Set[str]:
        return set(self.graph.nodes)
