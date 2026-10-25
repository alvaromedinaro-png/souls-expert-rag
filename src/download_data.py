"""Descarga la base documental del experto Souls desde Fandom y Wikipedia.

Cada página se guarda como Markdown en data/<juego>/<pagina>.md con una
cabecera de metadatos (título, juego, URL de origen y licencia).

Uso:
    python -m src.download_data            # descarga lo que falte
    python -m src.download_data --force    # vuelve a descargar todo
"""

import argparse
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "SoulsExpertRAG/1.0 (proyecto academico IA Generativa)"}

# Páginas de lore, personajes, jefes y mecánicas de cada wiki de Fandom.
FANDOM_PAGES = {
    "dark_souls": (
        "darksouls.fandom.com",
        [
            "Dark Souls", "Dark Souls II", "Dark Souls III",
            "Age of Fire", "Lords of Cinder", "Dark Soul", "Abyss",
            "Undead", "Hollowing", "Humanity (gameplay mechanic)",
            "Estus Flask (Dark Souls)", "Bonfire (Dark Souls)", "Covenant",
            "Primordial Serpents", "Furtive Pygmy",
            "Gwyn, Lord of Cinder", "Knight Artorias", "Great Grey Wolf Sif",
            "Seath the Scaleless", "Gravelord Nito", "Bed of Chaos",
            "Dragon Slayer Ornstein and Executioner Smough",
            "Kingseeker Frampt", "Darkstalker Kaathe", "Firelink Shrine",
            "Drangleic", "Vendrick", "Lothric", "Pontiff Sulyvahn",
            "The Nameless King", "Soul of Cinder", "The Ringed City",
            "Slave Knight Gael",
        ],
    ),
    "bloodborne": (
        "bloodborne.fandom.com",
        [
            "Bloodborne", "Yharnam", "The Healing Church", "Byrgenwerth",
            "Old Blood", "Paleblood", "Great Ones", "Hunter's Dream",
            "Insight", "Blood Echoes", "Beasthood", "Ashen Blood",
            "Blood Ministration", "Pthumerians", "Weapons", "The Doll",
            "Gehrman, the First Hunter", "Master Willem", "Moon Presence",
            "Father Gascoigne", "Cleric Beast", "Vicar Amelia",
            "Ebrietas, Daughter of the Cosmos", "Mergo", "Kos",
            "Orphan of Kos", "Lady Maria of the Astral Clocktower",
            "The Old Hunters", "Endings",
        ],
    ),
    "elden_ring": (
        "eldenring.fandom.com",
        [
            "Elden Ring: Shadow of the Erdtree", "Erdtree", "Golden Order",
            "Greater Will", "Outer God", "Two Fingers", "Great Runes",
            "The Shattering", "Tarnished", "Flask of Crimson Tears",
            "Ashes of War", "Stats and Attributes", "Roundtable Hold",
            "Torrent", "Melina", "Queen Marika the Eternal",
            "Radagon of the Golden Order", "Godfrey, First Elden Lord",
            "Godrick the Grafted", "Rennala, Queen of the Full Moon",
            "Ranni the Witch", "Starscourge Radahn",
            "Rykard, Lord of Blasphemy", "Morgott, the Omen King",
            "Mohg, Lord of Blood", "Malenia, Blade of Miquella",
            "Miquella of the Haligtree", "Maliketh, the Black Blade",
            "Fire Giant", "Dragonlord Placidusax", "Elden Beast",
            "Messmer the Impaler", "Endings",
        ],
    ),
}

# Artículos generales (desarrollo, lanzamiento, recepción) de Wikipedia.
WIKIPEDIA_PAGES = {
    "general": [
        "FromSoftware", "Hidetaka Miyazaki", "Soulslike", "Demon's Souls",
    ],
    "dark_souls": [
        "Dark Souls (video game)", "Dark Souls II", "Dark Souls III",
    ],
    "bloodborne": ["Bloodborne"],
    "elden_ring": ["Elden Ring", "Elden Ring: Shadow of the Erdtree"],
}

# Elementos de la página de Fandom que solo aportan ruido al RAG.
NOISE_SELECTORS = [
    "aside", "table", "figure", "script", "style", "sup.reference",
    ".navbox", ".toc", ".mw-editsection", ".gallery", ".wikia-gallery",
    ".reference", ".references", ".noprint", ".mw-empty-elt",
]
# Secciones finales que no contienen información del juego.
SKIP_SECTIONS = {
    "references", "external links", "see also", "notes", "gallery",
    "trivia", "videos", "navigation", "further reading", "footnotes",
}

# Notas editoriales de la wiki que no aportan contenido.
EDITOR_NOTES = re.compile(r"\s*\[(citation|verification|clarification) needed\]")


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fandom_to_markdown(html: str) -> str:
    """Convierte el HTML de una página de Fandom en Markdown limpio."""
    soup = BeautifulSoup(html, "html.parser")
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    lines, skipping = [], False
    for tag in soup.find_all(["h2", "h3", "h4", "p", "li", "blockquote"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
        text = EDITOR_NOTES.sub("", text).strip()
        if not text:
            continue
        if tag.name in ("h2", "h3", "h4"):
            skipping = text.lower() in SKIP_SECTIONS
            if not skipping:
                lines.append(f"\n{'#' * int(tag.name[1])} {text}\n")
        elif not skipping:
            # Evita duplicar texto anidado (listas dentro de <li>, citas con <p>).
            if tag.find_parent(["li", "blockquote"]):
                continue
            lines.append(f"- {text}" if tag.name == "li" else text)
    return "\n".join(lines)


def wikipedia_to_markdown(text: str) -> str:
    """Convierte el texto plano de Wikipedia (== Título ==) en Markdown."""
    lines, skipping = [], False
    for line in text.splitlines():
        match = re.match(r"^(=+)\s*(.*?)\s*=+$", line)
        if match:
            skipping = match.group(2).lower() in SKIP_SECTIONS
            if not skipping:
                lines.append(f"\n{'#' * len(match.group(1))} {match.group(2)}\n")
        elif not skipping and line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def fetch_fandom(host: str, title: str) -> tuple[str, str]:
    resp = requests.get(
        f"https://{host}/api.php",
        params={"action": "parse", "page": title, "prop": "text",
                "redirects": 1, "format": "json"},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["parse"]
    url = f"https://{host}/wiki/{data['title'].replace(' ', '_')}"
    return fandom_to_markdown(data["text"]["*"]), url


def fetch_wikipedia(title: str) -> tuple[str, str]:
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "redirects": 1, "titles": title, "format": "json"},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    url = f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
    return wikipedia_to_markdown(page["extract"]), url


def save(game: str, title: str, source: str, url: str, body: str) -> Path:
    path = DATA_DIR / game / f"{source}_{slugify(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"---\ntitle: {title}\ngame: {game}\nsource: {source}\n"
        f"url: {url}\nlicense: CC BY-SA\n---\n\n# {title}\n"
    )
    path.write_text(header + body.strip() + "\n", encoding="utf-8")
    return path


def main(force: bool = False) -> None:
    jobs = [(game, title, "fandom", host)
            for game, (host, titles) in FANDOM_PAGES.items() for title in titles]
    jobs += [(game, title, "wikipedia", None)
             for game, titles in WIKIPEDIA_PAGES.items() for title in titles]

    for game, title, source, host in jobs:
        target = DATA_DIR / game / f"{source}_{slugify(title)}.md"
        if target.exists() and not force:
            continue
        try:
            body, url = (fetch_fandom(host, title) if source == "fandom"
                         else fetch_wikipedia(title))
            path = save(game, title, source, url, body)
            print(f"[ok] {path.relative_to(DATA_DIR)} ({len(body):,} car.)")
        except Exception as exc:  # una página fallida no detiene el resto
            print(f"[error] {source}:{title} -> {exc}")
        time.sleep(0.5)  # cortesía con las APIs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="vuelve a descargar las páginas existentes")
    main(parser.parse_args().force)
