"""PII masking before outbound LLM calls (Presidio, regex recognizers).

Family financial data is sent to a *cloud* LLM via OpenRouter, so high-risk
identifiers must be stripped before they leave the machine. We use Presidio's
regex/context recognizers (phone, card, email, IBAN, IP) over a blank spaCy
tokenizer — **no NER model download**. PERSON-name NER is intentionally out of
scope (it would need ``ru_core_news_md``); the lean set covers the real leak
vectors for the diploma deadline.

Single entry points:
- :func:`mask_text` — anonymize one string.
- :func:`mask_messages` — anonymize the ``HumanMessage`` content of a prompt;
  our own system/AI messages and image parts are left untouched.

Wired in at the one LLM chokepoint — ``infrastructure/llm.get_chat_model`` —
so every outbound call is masked without per-node changes.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from langchain_core.messages import BaseMessage, HumanMessage
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Regex/context entities we strip before a cloud LLM. All detectable without NER.
_ENTITIES = ["PHONE_NUMBER", "CREDIT_CARD", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS"]

# Presidio works in one language pass; the identifiers above are language-neutral
# regexes, so we run the "en" recognizer set over the (often Russian) text.
_LANG = "en"


class _BlankSpacyNlpEngine(SpacyNlpEngine):
    """spaCy tokenizer with no statistical model — avoids any model download."""

    def __init__(self) -> None:
        super().__init__()
        # Base class types ``nlp`` as None until a model loads; we inject a
        # tokenizer-only pipeline (no NER, no download) directly.
        self.nlp = {_LANG: spacy.blank("en")}  # type: ignore[assignment]


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    analyzer = AnalyzerEngine(
        nlp_engine=_BlankSpacyNlpEngine(),
        supported_languages=[_LANG],
    )
    # Detect Russian numbers (8…, +7…) in addition to the default regions.
    analyzer.registry.add_recognizer(
        PhoneRecognizer(supported_regions=["RU", "US"], supported_language=_LANG)
    )
    return analyzer, AnonymizerEngine()  # type: ignore[no-untyped-call]


def mask_text(text: str) -> str:
    """Replace any detected PII with ``<ENTITY_TYPE>`` placeholders."""
    if not text:
        return text
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, entities=_ENTITIES, language=_LANG)
    if not results:
        return text
    return anonymizer.anonymize(
        text=text,
        # presidio-analyzer and -anonymizer ship distinct RecognizerResult
        # classes; they are structurally identical and the runtime accepts ours.
        analyzer_results=results,  # type: ignore[arg-type]
        operators={"DEFAULT": OperatorConfig("replace")},
    ).text


def mask_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Mask PII in user-supplied (``HumanMessage``) content only.

    System/AI messages are our own prompts and stay intact. Multimodal content
    (e.g. a receipt image + text) keeps non-text parts untouched.
    """
    return [_mask_message(m) for m in messages]


def _mask_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, HumanMessage):
        return message
    content = message.content
    if isinstance(content, str):
        return message.model_copy(update={"content": mask_text(content)})
    return message.model_copy(update={"content": [_mask_part(p) for p in content]})


def _mask_part(part: str | dict[str, object]) -> str | dict[str, object]:
    if isinstance(part, str):
        return mask_text(part)
    if part.get("type") == "text":
        text = part.get("text")
        if isinstance(text, str):
            return {**part, "text": mask_text(text)}
    return part
