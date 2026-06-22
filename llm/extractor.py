import json
import os
import re
import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are a clinical coding assistant for an ophthalmology practice.
Extract medical diagnoses and procedures from the transcript.
Return ONLY a valid JSON object — no explanation, no markdown:
{"clinical_terms": ["term1", "term2"], "laterality": "right|left|bilateral|unknown", "is_new_patient": true|false|null}

clinical_terms examples: "dry eye", "nuclear cataract", "open angle glaucoma", "diabetic retinopathy", "macular edema", "corneal ulcer", "retinal detachment", "floaters", "keratitis", "meibomian gland dysfunction"
laterality: extract from right eye/left eye/OD/OS/OU/bilateral mentions."""


def extract_clinical_concepts(transcript_segment: str) -> dict:
    if not API_KEY:
        return {"clinical_terms": [], "laterality": "unknown", "is_new_patient": None, "error": "GEMINI_API_KEY not set"}

    # Only keep doctor/attendant lines, cap at 1500 chars
    lines = [l.strip() for l in transcript_segment.split("\n")
             if l.strip().lower().startswith(("doctor:", "attendant:"))]
    condensed = "\n".join(lines) if lines else transcript_segment
    condensed = condensed[:1500]

    prompt = f"Ophthalmology transcript:\n{condensed}\n\nReturn JSON with clinical_terms, laterality, is_new_patient:"

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={API_KEY}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 512,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"[extractor] raw:\n{raw}\n---")
        result = _parse_response(raw)
        print(f"[extractor] parsed: {result}")
        return result
    except requests.exceptions.Timeout:
        return {"clinical_terms": [], "laterality": "unknown", "is_new_patient": None, "error": "timeout"}
    except Exception as e:
        print(f"[extractor] error: {e}")
        return {"clinical_terms": [], "laterality": "unknown", "is_new_patient": None, "error": str(e)}


def _parse_response(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    start = text.find("{")
    if start == -1:
        print(f"[extractor] no JSON found: {text[:200]}")
        return {"clinical_terms": [], "laterality": "unknown", "is_new_patient": None}

    json_text = text[start:]
    end = json_text.rfind("}") + 1

    if end > 0:
        try:
            parsed = json.loads(json_text[:end])
            return {
                "clinical_terms": parsed.get("clinical_terms", []),
                "laterality": parsed.get("laterality", "unknown"),
                "is_new_patient": parsed.get("is_new_patient", None),
            }
        except json.JSONDecodeError:
            pass

    # Fallback: extract terms via regex if JSON truncated
    terms_match = re.search(r'"clinical_terms"\s*:\s*\[([^\]]*)', json_text)
    if terms_match:
        terms = re.findall(r'"([^"]+)"', terms_match.group(1))
        lat_match = re.search(r'"laterality"\s*:\s*"([^"]+)"', json_text)
        laterality = lat_match.group(1).split("|")[0] if lat_match else "unknown"
        return {"clinical_terms": terms, "laterality": laterality, "is_new_patient": None}

    return {"clinical_terms": [], "laterality": "unknown", "is_new_patient": None}
