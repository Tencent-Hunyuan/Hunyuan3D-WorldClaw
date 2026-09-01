"""OpenAI-compatible API integration used by the non-Hunyuan workers.

Credentials are read from the environment or a user-owned auth.json file. No
secret is ever included in worker responses, manifests, or logs.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _default_home() -> Path:
    return Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))


def _strict_schema(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic JSON schemas for strict Responses gateways."""
    schema = json.loads(json.dumps(value))

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            for child in node.get("properties", {}).values():
                visit(child)
        if node.get("type") == "array" and "items" in node:
            visit(node["items"])
        for child in node.get("$defs", {}).values():
            visit(child)

    visit(schema)
    return schema


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    text_model: str
    vision_model: str
    image_model: str
    image_via_responses: bool = True
    timeout_seconds: float = 900.0
    validation_model: str = "gpt-5.6-sol"

    @classmethod
    def from_environment(cls) -> "OpenAIConfig":
        config_path = Path(
            _first_nonempty(
                os.getenv("WORLDCLAW_OPENAI_CONFIG_FILE"),
                os.getenv("OPENAI_CONFIG_FILE"),
                str(_default_home() / "config.toml"),
            )
        )
        auth_path = Path(
            _first_nonempty(
                os.getenv("WORLDCLAW_OPENAI_AUTH_FILE"),
                os.getenv("OPENAI_AUTH_FILE"),
                str(_default_home() / "auth.json"),
            )
        )

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key and auth_path.is_file():
            raw = json.loads(auth_path.read_text(encoding="utf-8"))
            api_key = str(raw.get("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is missing; set OPENAI_API_KEY or configure "
                "WORLDCLAW_OPENAI_AUTH_FILE"
            )

        provider_name = ""
        configured_model = ""
        configured_base_url = ""
        if config_path.is_file():
            try:
                import tomllib
            except ModuleNotFoundError as exc:  # pragma: no cover - Python <3.11
                raise RuntimeError("tomllib is required to read config.toml") from exc
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            provider_name = str(config.get("model_provider", "")).strip()
            configured_model = str(config.get("model", "")).strip()
            providers = config.get("model_providers", {})
            provider = providers.get(provider_name, {}) if isinstance(providers, dict) else {}
            if isinstance(provider, dict):
                configured_base_url = str(provider.get("base_url", "")).strip()

        base_url = _first_nonempty(os.getenv("OPENAI_BASE_URL"), configured_base_url)
        if not base_url:
            base_url = "https://api.openai.com/v1"
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1") and not os.getenv("OPENAI_BASE_URL"):
            base_url += "/v1"

        text_model = _first_nonempty(
            os.getenv("OPENAI_TEXT_MODEL"), configured_model, "gpt-5"
        )
        vision_model = _first_nonempty(
            os.getenv("OPENAI_VISION_MODEL"), configured_model, text_model
        )
        validation_model = _first_nonempty(
            os.getenv("OPENAI_VALIDATION_MODEL"), "gpt-5.6-sol"
        )
        configured_image_model = os.getenv("OPENAI_IMAGE_MODEL", "").strip()
        image_model = _first_nonempty(configured_image_model, text_model)
        return cls(
            api_key=api_key,
            base_url=base_url,
            text_model=text_model or "gpt-5",
            vision_model=vision_model or text_model or "gpt-5",
            image_model=image_model or "gpt-5",
            validation_model=validation_model or "gpt-5.6-sol",
            image_via_responses=not bool(configured_image_model),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "900")),
        )

    def public_dict(self) -> dict[str, Any]:
        """Return auditable configuration without exposing credentials."""
        return {
            "provider": "openai-compatible",
            "base_url": self.base_url,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "image_model": self.image_model,
            "validation_model": self.validation_model,
            "image_via_responses": self.image_via_responses,
            "api_key_configured": bool(self.api_key),
        }


class OpenAIClient:
    def __init__(self, config: OpenAIConfig | None = None):
        self.config = config or OpenAIConfig.from_environment()
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised on remote env
            raise RuntimeError(
                "openai package is required for OpenAI workers; install the live extra"
            ) from exc
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            default_headers={"User-Agent": "WorldClaw/1.0"},
        )

    @staticmethod
    def _parse_json_output(response: Any) -> dict[str, Any]:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return OpenAIClient._decode_json_text(output_text)
        # Compatibility with OpenAI-compatible gateways that return the raw
        # Chat Completions shape even when the Responses wire API is requested.
        data = response if isinstance(response, dict) else response.model_dump()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content)
            return OpenAIClient._decode_json_text(content)
        raise RuntimeError("OpenAI response did not contain structured output text")

    @staticmethod
    def _decode_json_text(value: str) -> dict[str, Any]:
        text = value.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text.strip())

    def json_response(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self.client.responses.create(
                model=model,
                instructions=system,
                input=user,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "worldclaw_response",
                        "strict": True,
                        "schema": _strict_schema(schema),
                    }
                },
            )
        except Exception as exc:
            # A few OpenAI-compatible gateways reject complex nested schemas
            # with a 5xx even though ordinary Responses calls work. Preserve
            # local Pydantic validation by asking for JSON text as a fallback.
            status = getattr(exc, "status_code", None)
            if status is not None and int(status) < 500:
                raise
            try:
                response = self.client.responses.create(
                    model=model,
                    instructions=system + "\nReturn one valid JSON object only; no Markdown.",
                    input=user,
                )
            except Exception:
                return self._chat_json_response(model=model, system=system, user=user, schema=schema)
        return self._parse_json_output(response)

    def _chat_json_response(
        self, *, model: str, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            # The gateway accepts JSON mode on Chat Completions but may return
            # 5xx for nested strict schemas.  Pydantic validates the result.
            response_format={"type": "json_object"},
        )
        return self._parse_json_output(response)

    def vision_json(
        self,
        *,
        system: str,
        user: str,
        image_paths: Iterable[Path],
        schema: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        image_paths = list(image_paths)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user}]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            })
        input_value = [{"role": "user", "content": content}]
        selected_model = model or self.config.vision_model
        try:
            response = self.client.responses.create(
                model=selected_model,
                instructions=system,
                input=input_value,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "worldclaw_vision_response",
                        "strict": True,
                        "schema": _strict_schema(schema),
                    }
                },
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and int(status) < 500:
                raise
            try:
                response = self.client.responses.create(
                    model=selected_model,
                    instructions=system + "\nReturn one valid JSON object only; no Markdown.",
                    input=input_value,
                )
            except Exception:
                chat_content: list[dict[str, Any]] = [{"type": "text", "text": user}]
                for path in image_paths:
                    mime = mimetypes.guess_type(path.name)[0] or "image/png"
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    chat_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    })
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": chat_content}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
        return self._parse_json_output(response)

    def generate_image(
        self,
        *,
        prompt: str,
        output_path: Path,
        conditioning_image: Path | None = None,
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.image_via_responses:
            return self._generate_image_via_responses(
                prompt=prompt,
                output_path=output_path,
                conditioning_image=conditioning_image,
            )
        if conditioning_image is None:
            try:
                result = self.client.images.generate(
                    model=self.config.image_model,
                    prompt=prompt,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json",
                )
            except Exception as exc:
                # Some OpenAI-compatible gateways expose image generation only
                # as a Responses tool even when an image model is configured.
                if "model_not_found" not in str(exc).lower() and "not supported" not in str(exc).lower():
                    raise
                return self._generate_image_via_responses(
                    prompt=prompt, output_path=output_path, conditioning_image=None
                )
        else:
            with conditioning_image.open("rb") as stream:
                result = self.client.images.edit(
                    model=self.config.image_model,
                    image=stream,
                    prompt=prompt,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json",
                )
        item = result.data[0]
        encoded = getattr(item, "b64_json", None)
        if encoded:
            output_path.write_bytes(base64.b64decode(encoded))
        else:
            url = getattr(item, "url", None)
            if not url:
                raise RuntimeError("OpenAI image response had neither b64_json nor url")
            with urllib.request.urlopen(url, timeout=self.config.timeout_seconds) as response:
                output_path.write_bytes(response.read())
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"OpenAI image output is empty: {output_path}")
        return {"path": str(output_path), "model": self.config.image_model}

    def _generate_image_via_responses(
        self,
        *,
        prompt: str,
        output_path: Path,
        conditioning_image: Path | None,
    ) -> dict[str, Any]:
        input_value: Any = prompt
        if conditioning_image is not None:
            mime = mimetypes.guess_type(conditioning_image.name)[0] or "image/png"
            encoded = base64.b64encode(conditioning_image.read_bytes()).decode("ascii")
            input_value = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
                ],
            }]
        response = self.client.responses.create(
            model=self.config.text_model,
            input=input_value,
            tools=[{"type": "image_generation"}],
        )
        result = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "image_generation_call":
                result = getattr(item, "result", None)
                if result:
                    break
        if result is None:
            data = response.model_dump() if hasattr(response, "model_dump") else response
            for item in data.get("output", []):
                if item.get("type") == "image_generation_call":
                    result = item.get("result")
                    if result:
                        break
        if not result:
            raise RuntimeError("Responses image_generation returned no image result")
        output_path.write_bytes(base64.b64decode(result))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"OpenAI Responses image output is empty: {output_path}")
        return {"path": str(output_path), "model": self.config.text_model, "via": "responses_tool"}
