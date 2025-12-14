# app/llm_client.py
import os
import json
import re
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("llm_client")


class LLMError(Exception):
    pass


class LLMClient:
    """
    Pluggable LLM client supporting:
      - SAP AI Core (Generative AI Hub)
      - Local HTTP LLM endpoint
    """

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "sap_aicore").lower()
       
        # SAP AI Core config
        self.token_url = os.environ.get("AI_CORE_TOKEN_URL")
        #self.token_url = 'https://hclbuild-g03o2ijo.authentication.eu10.hana.ondemand.com/oauth/token'

        print(f"AI_CORE_TOKEN_URL: {self.token_url}")
        self.client_id = os.environ.get("AI_CORE_CLIENT_ID")
        #self.client_id = 'sb-7a6a197c-b8cf-4dc5-a5fb-7b22699dd76e!b227908|aicore!b540'
        print(f"AI_CORE_CLIENT_ID: {self.client_id}")
        self.client_secret = os.environ.get("AI_CORE_CLIENT_SECRET")  
        #self.client_secret = 'c2b56fd1-225c-40d8-a2fb-955582fcf213$Uat6AcTYsnWmWw_LxlqWv3EgJEhH1UPqj8ncVwbsJLE='   
        print(f"AI_CORE_CLIENT_SECRET: {self.client_secret}")
        self.inference_url = os.environ.get("AI_CORE_INFERENCE_URL")
        #self.inference_url = 'https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d24683bad313b157/chat/completions?api-version=2023-05-15'
        print(f"AI_CORE_INFERENCE_URL: {self.inference_url}")
        self.model_name = os.environ.get("AI_CORE_MODEL_NAME")
        print(f"AI_CORE_MODEL_NAME: {self.model_name}")
        # Local LLM
        self.local_url = os.environ.get("LOCAL_LLM_URL")
        print(f"LOCAL_LLM_URL: {self.local_url}")       
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "800"))
        print(f"LLM_MAX_TOKENS: {self.max_tokens}")
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
        print(f"LLM_TEMPERATURE: {self.temperature}")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        if self.provider == "sap_aicore":
            return self._call_sap_aicore(prompt)
        elif self.provider in ("local", "ollama", "vllm"):
            return self._call_local(prompt)
        else:
            raise LLMError(f"Unknown LLM_PROVIDER: {self.provider}")

    # ------------------------------------------------------------------
    # SAP AI Core
    # ------------------------------------------------------------------
    def _get_access_token(self) -> str:
        try:
            resp = requests.post(
                self.token_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["access_token"]
        except Exception as e:
            logger.exception("Failed to fetch SAP AI Core token")
            raise LLMError("SAP AI Core authentication failed") from e

    def _call_sap_aicore(self, prompt: str) -> Dict[str, Any]:
        if not all([self.token_url, self.client_id, self.client_secret, self.inference_url]):
            raise LLMError("SAP AI Core configuration missing")

        token = self._get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an SAP short dump analysis expert."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            r = requests.post(
                self.inference_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            raw = r.json()

            text = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            parsed = self._try_parse_json(text)
            return {"text": text, "json": parsed, "raw": raw}

        except Exception as e:
            logger.exception("SAP AI Core inference failed")
            raise LLMError(str(e)) from e

    # ------------------------------------------------------------------
    # Local LLM
    # ------------------------------------------------------------------
    def _call_local(self, prompt: str) -> Dict[str, Any]:
        if not self.local_url:
            raise LLMError("LOCAL_LLM_URL not set")

        try:
            r = requests.post(self.local_url, json={"prompt": prompt}, timeout=60)
            r.raise_for_status()
            j = r.json()
            text = j.get("result") or j.get("text") or j.get("output") or json.dumps(j)
            parsed = self._try_parse_json(text)
            return {"text": text, "json": parsed, "raw": j}
        except Exception as e:
            logger.exception("Local LLM call failed")
            raise LLMError(str(e)) from e

    # ------------------------------------------------------------------
    # JSON extraction helper
    # ------------------------------------------------------------------
    def _try_parse_json(self, text: str):
        if not text:
            return None

        s = text.strip()
        s = re.sub(r"```(?:json)?\n", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n```$", "", s)

        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(s[first:last + 1])
            except Exception:
                pass

        try:
            return json.loads(s)
        except Exception:
            return None
