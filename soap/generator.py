import os
import re
import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are a clinical documentation assistant for an ophthalmology practice.
Generate a structured SOAP note from the visit transcript and accepted ICD-10/CPT codes.

Format exactly as:

SUBJECTIVE:
[Chief complaint and patient-reported symptoms]

OBJECTIVE:
[VA, IOP, slit lamp, fundus, OCT findings mentioned by doctor]

ASSESSMENT:
[Diagnoses with ICD-10 codes]

PLAN:
[Treatment, medications, procedures, follow-up. Include CPT codes]

Use standard abbreviations: OD, OS, OU, VA, IOP, SLE. Be concise. Only include findings from the transcript."""


def generate_soap_note(full_transcript: str, accepted_codes: list[dict]) -> str:
    if not API_KEY:
        return "Error: GEMINI_API_KEY not set in environment."

    codes_text = _format_codes(accepted_codes)
    transcript_trimmed = full_transcript[:4000]

    prompt = f"""Transcript:
{transcript_trimmed}

Accepted codes:
{codes_text}

Write the SOAP note:"""

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={API_KEY}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return raw.strip()
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error generating SOAP note: {e}"


def _format_codes(accepted_codes: list[dict]) -> str:
    if not accepted_codes:
        return "No codes accepted."
    lines = []
    for entry in accepted_codes:
        lines.append(f"ICD-10: {entry['icd_code']} - {entry['icd_description']}")
        cpts = ", ".join(f"CPT {c['code']} ({c['description']})" for c in entry.get("cpt_codes", []))
        if cpts:
            lines.append(f"  → {cpts}")
    return "\n".join(lines)
