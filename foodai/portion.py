"""Portion estimation with Gemini Vision (optional). Falls back to a default weight if anything goes wrong."""
import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from PIL import Image

PORTION_PROMPT = """You are a nutrition expert estimating food portions from a photo.

Food identified: "{food_name}"

Look at the image and estimate the total weight of this food in grams, based on typical plate size,
food density, and visible volume. Respond with ONLY a JSON object in this exact format, no extra text:

{{"estimated_grams": <number>, "confidence": "<low|medium|high>", "reasoning": "<one short sentence>"}}
"""

DEFAULT_MODEL = "gemini-3.8-flash"
HARD_DEADLINE_S = 25  # the request NEVER waits longer than this for Gemini
_pool = ThreadPoolExecutor(max_workers=2)

# --- IPv6 workaround -------------------------------------------------------------------
# On some networks IPv6 to Google times out while IPv4 works instantly. Python then waits for
# every IPv6 address to fail (minutes!) before trying IPv4. For the Gemini host only, use IPv4.
GEMINI_HOST = "generativelanguage.googleapis.com"


def _install_ipv4_preference():
    if getattr(socket.getaddrinfo, "_ipv4_patched", False):
        return
    original = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        results = original(host, *args, **kwargs)
        if host == GEMINI_HOST:
            v4 = [r for r in results if r[0] == socket.AF_INET]
            if v4:
                return v4
        return results

    getaddrinfo._ipv4_patched = True
    socket.getaddrinfo = getaddrinfo


_install_ipv4_preference()


def _clean_env(name, default=""):
    """Read an env var, tolerating blanks, spaces and quotes (e.g. GEMINI_MODEL="" in .env)."""
    value = (os.environ.get(name) or "").strip().strip('"').strip("'").strip()
    return value or default


def _call_gemini(api_key, model, food_name, image):
    from google import genai
    from google.genai import types

    http_kwargs = {"timeout": 20000}  # milliseconds
    try:
        http_kwargs["retry_options"] = types.HttpRetryOptions(attempts=1)  # no silent retries
    except (AttributeError, TypeError):
        pass
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http_kwargs))
    resp = client.models.generate_content(
        model=model,
        contents=[PORTION_PROMPT.format(food_name=food_name), image],
    )
    return resp.text


def estimate_portion_grams(image: Image.Image, food_name: str, default_grams: float = 150):
    api_key = _clean_env("GOOGLE_API_KEY")
    if not api_key:
        return {"estimated_grams": default_grams, "confidence": "none",
                "reasoning": "GOOGLE_API_KEY not set; using default portion."}

    model = _clean_env("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        future = _pool.submit(_call_gemini, api_key, model, food_name, image.convert("RGB"))
        text = future.result(timeout=HARD_DEADLINE_S)
        text = re.sub(r"^```json|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"[\d.]+", text)
            grams = float(m.group()) if m else default_grams
            return {"estimated_grams": grams, "confidence": "low", "reasoning": text}
    except FutureTimeout:
        return {"estimated_grams": default_grams, "confidence": "none",
                "reasoning": f"Gemini did not answer within {HARD_DEADLINE_S}s (check internet / model '{model}'); using default portion."}
    except Exception as e:  # network / quota / bad model name: never break the whole pipeline
        return {"estimated_grams": default_grams, "confidence": "none",
                "reasoning": f"Gemini call failed with model '{model}' ({type(e).__name__}: {e}); using default portion."}
