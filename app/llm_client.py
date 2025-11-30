# app/llm_client.py
import os
import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger("llm_client")

class LLMError(Exception):
    pass

class LLMClient:
    """
    Pluggable LLM client supporting:
      - OpenAI (openai package)
      - Local HTTP LLM endpoint (simple POST -> { "result": ... })
    The analyze(prompt) method returns a dict:
      {
        "text": <raw text>,
        "json": <parsed JSON if parseable>,
        "raw": <provider raw response>
      }
    """
    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.local_url = os.environ.get("LOCAL_LLM_URL")  # e.g. http://127.0.0.1:11434/generate
        # simple tunables
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "800"))
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

    def analyze(self, prompt: str) -> Dict[str, Any]:
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider in ("local", "ollama", "vllm"):
            return self._call_local(prompt)
        else:
            raise LLMError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        try:
            import openai
        except Exception as e:
            raise LLMError("openai library not available. Install openai in requirements.") from e

        if not self.openai_api_key:
            raise LLMError("OPENAI_API_KEY not set")

        openai.api_key = self.openai_api_key

        try:
            # Using ChatCompletion for broad compatibility
            resp = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            text = resp["choices"][0]["message"]["content"]
            parsed = self._try_parse_json(text)
            return {"text": text, "json": parsed, "raw": resp}
        except Exception as e:
            logger.exception("OpenAI call failed")
            raise LLMError(str(e)) from e

    def _call_local(self, prompt: str) -> Dict[str, Any]:
        if not self.local_url:
            raise LLMError("LOCAL_LLM_URL not set for local provider")
        import requests
        try:
            r = requests.post(self.local_url, json={"prompt": prompt}, timeout=60)
            r.raise_for_status()
            j = r.json()
            # support common response wrappers
            text = j.get("result") or j.get("text") or j.get("output") or json.dumps(j)
            parsed = self._try_parse_json(text)
            return {"text": text, "json": parsed, "raw": j}
        except Exception as e:
            logger.exception("Local LLM call failed")
            raise LLMError(str(e)) from e

    def _try_parse_json(self, text: str):
        """
        Attempts to extract JSON embedded in text. Returns Python object or None.
        This is conservative and avoids eval(); supports single JSON object or JSON block.
        """
        if not text:
            return None
        # common technique: find first '{' and last '}' and try json.loads
        # but be cautious: returns None on failure
        # remove surrounding markdown ```json ... ```
        # strip leading/trailing whitespace
        s = text.strip()
        # remove triple-backtick blocks
        s = re.sub(r"```(?:json)?\n", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n```$", "", s)
        # find first { and last }
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = s[first:last+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # fallback: try to parse entire string
        try:
            return json.loads(s)
        except Exception:
            return None
