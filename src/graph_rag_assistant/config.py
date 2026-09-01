from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "apollo_docs"

GRAPH_ENTITIES = {
    "Person": {
        "Neil Armstrong",
        "Buzz Aldrin",
        "Michael Collins",
        "John Young",
        "Charles Duke",
        "Gene Cernan",
        "Harrison Schmitt",
        "Alan Shepard",
        "Edgar Mitchell",
        "Charlie Conrad",
        "Alan Bean",
    },
    "Mission": {
        "Apollo 11",
        "Apollo 12",
        "Apollo 13",
        "Apollo 14",
        "Apollo 15",
        "Apollo 16",
        "Apollo 17",
    },
    "Location": {
        "Moon",
        "Sea of Tranquility",
        "Fra Mauro",
        "Hadley-Apennine",
        "Descartes Highlands",
        "Taurus-Littrow",
        "Ocean of Storms",
    },
    "Institution": {"NASA", "Kennedy Space Center", "Jet Propulsion Laboratory"},
}
