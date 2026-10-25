"""
Explanatory agent on Azure OpenAI: turns the FGS, thermal and ontology results into a readable
report, and answers questions about the report.

Configuration in .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
AZURE_OPENAI_API_VERSION. Without configuration it returns a plain-text summary of the same data,
labelled as such.
"""

import json
from functools import lru_cache

import config
from agent.prompts import CHAT_PROMPT, SYSTEM_VET_PROMPT


class AgentError(RuntimeError):
    """Error while calling Azure OpenAI."""


@lru_cache(maxsize=1)
def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=config.AZURE_OPENAI_ENDPOINT, api_key=config.AZURE_OPENAI_API_KEY,
                       api_version=config.AZURE_OPENAI_API_VERSION)


def _complete(messages: list[dict]) -> str:
    # no temperature: gpt-6-luna only accepts the default value
    try:
        response = _client().chat.completions.create(model=config.AZURE_OPENAI_DEPLOYMENT, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        raise AgentError(f"Azure OpenAI: {type(e).__name__}: {e}") from e


def _facts(cat_name: str, fgs: dict, thermal: dict, ontology: dict) -> str:
    rules = "\n".join(f"  - {r['swrl']}" for r in ontology["fired_rules"]) or "  - none"
    return (f"Patient: {cat_name}\n"
            f"- FGS: Ears={fgs['ear']}, Eyes={fgs['eye']}, Muzzle={fgs['muzzle']} (Total: {fgs['total']}/6)\n"
            f"- Thermal: {thermal['temperature']} °C estimated (surface {thermal['surface_max_c']} °C), "
            f"status {thermal['thermal_status']}\n"
            f"- Ontology ({ontology['reasoner']}): status {ontology['primary_status']} — {ontology['explanation']}\n"
            f"- Fired SWRL rules:\n{rules}")


def _summary(cat_name: str, fgs: dict, thermal: dict, ontology: dict) -> str:
    return ("Automatic summary (Azure OpenAI not available):\n\n"
            f"**Overall outcome**: {ontology['primary_status']} — {ontology['explanation']}.\n"
            f"**Facial evidence (FGS)**: total {fgs['total']}/6 "
            f"(ears {fgs['ear']}, eyes {fgs['eye']}, muzzle {fgs['muzzle']}).\n"
            f"**Thermal evidence**: {thermal['temperature']} °C estimated, status {thermal['thermal_status']}.\n"
            f"**Ontological inference**: "
            + ("; ".join(r["swrl"] for r in ontology["fired_rules"]) or "no rule fired") + ".")


def generate_multimodal_explanation(cat_name: str, fgs: dict, thermal: dict, ontology: dict) -> dict:
    """
    Integrated report. :return: {"text": str, "engine": "azure:<deployment>" | "summary", "error": str|None}
    """
    if not config.azure_configured():
        return {"text": _summary(cat_name, fgs, thermal, ontology), "engine": "summary", "error": None}
    try:
        text = _complete([{"role": "system", "content": SYSTEM_VET_PROMPT},
                          {"role": "user", "content": _facts(cat_name, fgs, thermal, ontology)}])
        return {"text": text, "engine": f"azure:{config.AZURE_OPENAI_DEPLOYMENT}", "error": None}
    except AgentError as e:
        return {"text": _summary(cat_name, fgs, thermal, ontology), "engine": "summary", "error": str(e)}


def chat(question: str, context: dict, history: list[dict] | None = None) -> dict:
    """Free question about the report. :return: {"answer": str, "engine": str}"""
    if not config.azure_configured():
        return {"answer": "The chat needs Azure OpenAI: set the AZURE_OPENAI_* variables in the .env file.",
                "engine": "none"}
    messages = [{"role": "system", "content": CHAT_PROMPT.format(context=json.dumps(context, ensure_ascii=False))}]
    for m in (history or [])[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": str(m["content"])[:2000]})
    messages.append({"role": "user", "content": question[:2000]})
    return {"answer": _complete(messages), "engine": f"azure:{config.AZURE_OPENAI_DEPLOYMENT}"}
