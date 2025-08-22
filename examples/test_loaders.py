import os
import traceback
import sys
import logging

logging.basicConfig(level=logging.ERROR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INPUT_PDF_DIR = os.path.join(REPO_ROOT, "inputdata_development_paper_set")

# Ensure 'src' is on sys.path so 'publications' package is importable when running directly
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from publications import (
    load_publications_from_folder,
    load_publications_from_hf_ard,
    load_publications_from_arxiv_ids,
)
from publications.utils import parse_arxiv_id_from_filename, extract_arxiv_id_from_url

import re


def test_load_from_folder():
    print("=== Test: load_publications_from_folder ===")
    pubs = load_publications_from_folder(INPUT_PDF_DIR)
    print(f"Loaded {len(pubs)} publications from folder")
    for p in pubs[:3]:
        refs_len = len(p.references) if p.references else 0
        print(f"- {p.title} | authors={len(p.authors)} | date={p.date_published} | abstract_len={len(p.abstract)} | text_len={len(p.text)} | refs_len={refs_len}")


def test_load_from_hf():
    print("=== Test: load_publications_from_hf_ard ===")
    try:
        pubs = load_publications_from_hf_ard(
            repo_id="StampyAI/alignment-research-dataset",
            local_dir=os.path.join(REPO_ROOT, "alignment-research-dataset"),
            dedupe=True,
        )
        print(f"Loaded {len(pubs)} publications from HF ARD")
        for p in pubs[:3]:
            refs_len = len(p.references) if p.references else 0
            print(f"- {p.title} | authors={p.authors} | date={p.date_published} | text_len={len(p.text)} | refs_len={refs_len}")
    except Exception:
        print("HF ARD test failed:")
        traceback.print_exc()


def _read_arxiv_ids_from_directory_file(directory_file: str) -> list[str]:
    """Read arXiv IDs from a directory.txt-style file.

    The file may contain raw IDs (e.g., 2311.07590v4), arXiv URLs, or filenames.
    Extract any valid modern-format arXiv IDs and preserve order without duplicates.
    """
    ids: list[str] = []
    seen = set()
    id_re = re.compile(r"(?P<id>\d{4}\.\d{5}(?:v\d+)?)")
    if not os.path.isfile(directory_file):
        raise FileNotFoundError(f"directory file not found: {directory_file}")
    with open(directory_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Try URL first
            arxiv_id = extract_arxiv_id_from_url(s)
            if not arxiv_id:
                # Try filename-like
                arxiv_id = parse_arxiv_id_from_filename(s)
            if not arxiv_id:
                # Fallback: regex anywhere in the line
                m = id_re.search(s)
                arxiv_id = m.group("id") if m else None
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                ids.append(arxiv_id)
    return ids


def test_load_from_arxiv_ids():
    print("=== Test: load_publications_from_arxiv_ids ===")
    # Read arXiv IDs from directory.txt in the input data folder
    directory_file = os.path.join(INPUT_PDF_DIR, "directory.txt")
    ids = _read_arxiv_ids_from_directory_file(directory_file)
    if not ids:
        raise RuntimeError("No arXiv IDs found in directory.txt")
    # Keep the test lightweight
    ids = ids[:3]
    pubs = load_publications_from_arxiv_ids(
        ids, download_pdf=True, pdf_dir=os.path.join(REPO_ROOT, "arxiv_pdfs_test")
    )
    print(f"Loaded {len(pubs)} publications from arXiv IDs: {ids}")
    for p in pubs:
        refs_len = len(p.references) if p.references else 0
        print(f"- {p.title} | authors={len(p.authors)} | date={p.date_published} | abstract_len={len(p.abstract)} | text_len={len(p.text)} | refs_len={refs_len}")


if __name__ == "__main__":
    # Run tests sequentially; let exceptions stop the run to surface issues
    test_load_from_folder()
    test_load_from_hf()
    test_load_from_arxiv_ids()
