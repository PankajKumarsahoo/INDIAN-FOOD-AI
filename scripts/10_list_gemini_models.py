"""Step-by-step Gemini connection test. Run:  python scripts/11_test_gemini.py"""
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodai.config import PROJECT_ROOT  # noqa: E402  (also loads .env)
from foodai.portion import DEFAULT_MODEL, _clean_env  # noqa: E402

HOST = "generativelanguage.googleapis.com"


def main():
    env_file = PROJECT_ROOT / ".env"
    key = _clean_env("GOOGLE_API_KEY")
    model = _clean_env("GEMINI_MODEL", DEFAULT_MODEL)

    print("1) Settings")
    print("   .env file :", env_file, "| exists:", env_file.exists())
    print("   API key   :", f"set ({len(key)} chars, starts with {key[:4]}...)" if key else "NOT SET")
    print("   Model     :", model, "(default used)" if not _clean_env("GEMINI_MODEL") else "")
    if " " in model:
        print("   !! The model name contains spaces. It must look like: gemini-xxxx-flash")
    if not key:
        raise SystemExit("   Put GOOGLE_API_KEY=... in the .env file shown above, save, and run again.")

    print("\n2) Network: can this computer reach Google?")
    try:
        infos = socket.getaddrinfo(HOST, 443, proto=socket.IPPROTO_TCP)
    except OSError as e:
        raise SystemExit(f"   DNS lookup failed: {e}\n   -> Check your internet / VPN / proxy / firewall.")
    seen = set()
    for fam, _, _, _, addr in infos:
        if addr[0] in seen:
            continue
        seen.add(addr[0])
        label = "IPv6" if fam == socket.AF_INET6 else "IPv4"
        t = time.time()
        try:
            with socket.create_connection(addr[:2], timeout=8):
                print(f"   {label} {addr[0]}: connected in {time.time() - t:.2f}s")
        except OSError as e:
            print(f"   {label} {addr[0]}: FAILED after {time.time() - t:.1f}s ({e})")

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000))

    print("\n3) Text-only call (30 s limit)")
    t = time.time()
    try:
        r = client.models.generate_content(model=model, contents="Reply with the single word: ok")
        print(f"   OK in {time.time() - t:.1f}s -> {r.text.strip()[:60]!r}")
    except Exception as e:
        print(f"   FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {str(e)[:300]}")

    print("\n4) Image call (30 s limit)")
    t = time.time()
    try:
        r = client.models.generate_content(
            model=model, contents=["What colour is this image? One word.", Image.new("RGB", (64, 64), "red")])
        print(f"   OK in {time.time() - t:.1f}s -> {r.text.strip()[:60]!r}")
    except Exception as e:
        print(f"   FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {str(e)[:300]}")

    print("\nSend me this whole output if anything above says FAILED or took more than ~10 s.")


if __name__ == "__main__":
    main()