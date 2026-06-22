import sqlite3
import csv
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ophtha.db"
CSV_PATH = Path(__file__).parent.parent / "data" / "ophtha_codes.csv"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and seed from CSV if DB doesn't exist."""
    if DB_PATH.exists():
        return

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS icd_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icd_code TEXT UNIQUE NOT NULL,
            icd_description TEXT NOT NULL,
            keywords TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cpt_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icd_code TEXT NOT NULL,
            cpt_code TEXT NOT NULL,
            cpt_description TEXT NOT NULL,
            FOREIGN KEY (icd_code) REFERENCES icd_codes(icd_code)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_icd ON cpt_mappings(icd_code)")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(
                "INSERT OR IGNORE INTO icd_codes (icd_code, icd_description, keywords) VALUES (?, ?, ?)",
                (row["icd_code"], row["icd_description"], row["keywords"]),
            )
            cpt_codes = row["cpt_codes"].split("|")
            cpt_descs = row["cpt_descriptions"].split("|")
            for code, desc in zip(cpt_codes, cpt_descs):
                cur.execute(
                    "INSERT INTO cpt_mappings (icd_code, cpt_code, cpt_description) VALUES (?, ?, ?)",
                    (row["icd_code"], code.strip(), desc.strip()),
                )

    conn.commit()
    conn.close()


def search_by_keywords(terms: list[str], limit: int = 5) -> list[dict]:
    """
    Given a list of clinical terms extracted by the LLM, find matching ICD codes.
    Uses keyword overlap scoring — no fuzzy lib needed.
    """
    if not terms:
        return []

    conn = get_connection()
    cur = conn.cursor()

    scores = {}
    for term in terms:
        term_lower = term.lower()
        cur.execute("SELECT icd_code, icd_description, keywords FROM icd_codes")
        for row in cur.fetchall():
            keyword_list = row["keywords"].lower().split()
            desc_lower = row["icd_description"].lower()
            score = 0
            for word in term_lower.split():
                if any(word in kw for kw in keyword_list):
                    score += 2
                if word in desc_lower:
                    score += 1
            if score > 0:
                if row["icd_code"] not in scores or scores[row["icd_code"]]["score"] < score:
                    scores[row["icd_code"]] = {
                        "icd_code": row["icd_code"],
                        "icd_description": row["icd_description"],
                        "score": score,
                    }

    sorted_matches = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:limit]

    results = []
    for match in sorted_matches:
        cur.execute(
            "SELECT cpt_code, cpt_description FROM cpt_mappings WHERE icd_code = ?",
            (match["icd_code"],),
        )
        cpt_rows = cur.fetchall()
        results.append({
            "icd_code": match["icd_code"],
            "icd_description": match["icd_description"],
            "score": match["score"],
            "cpt_codes": [{"code": r["cpt_code"], "description": r["cpt_description"]} for r in cpt_rows],
        })

    conn.close()
    return results


def get_all_icd_codes() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT icd_code, icd_description FROM icd_codes ORDER BY icd_code")
    rows = [{"icd_code": r["icd_code"], "icd_description": r["icd_description"]} for r in cur.fetchall()]
    conn.close()
    return rows
