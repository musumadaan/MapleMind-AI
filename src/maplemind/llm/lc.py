"""LangChain integration for generation.

Every model call in the agents goes through an LCEL chain:  ChatPromptTemplate | chat_model
  - provider "ollama"      -> langchain_ollama.ChatOllama          (real LangChain integration package)
  - provider "openai_compat" -> MapleChatModel(OpenAICompatLLM)
  - provider "hf"          -> MapleChatModel(HFLocalLLM, optional LoRA adapter)
  - provider "mock"        -> MapleChatModel(MockLLM)              (offline test double)
MapleChatModel is a LangChain BaseChatModel, so callbacks, tracing (LangSmith), .bind(), .batch() and streaming
wrappers work the same for all providers. Token usage is returned in AIMessage.usage_metadata.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .providers import BaseLLM, build_llm

_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


class MapleChatModel(BaseChatModel):
    """Adapts a MapleMind provider to LangChain's chat-model interface."""
    provider: Any = None

    @property
    def _llm_type(self) -> str:
        return f"maplemind-{getattr(self.provider, 'name', 'llm')}"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        msgs = [{"role": _ROLE.get(m.type, "user"), "content": m.content} for m in messages]
        hint = kwargs.pop("hint", None)
        r = self.provider.generate(msgs, hint=hint, **{k: v for k, v in kwargs.items() if k in ("temperature", "top_p", "max_new_tokens", "seed")})
        msg = AIMessage(content=r.text, usage_metadata={"input_tokens": r.tokens_in, "output_tokens": r.tokens_out,
                                                         "total_tokens": r.tokens_in + r.tokens_out},
                        response_metadata={"model": r.model})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def build_chat_model(cfg: dict, use_adapter: bool = False, provider: BaseLLM | None = None) -> BaseChatModel:
    if cfg.get("provider") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg["model"], base_url=cfg.get("base_url", "http://localhost:11434"),
                          temperature=cfg.get("temperature", 0.2), top_p=cfg.get("top_p", 0.9),
                          num_predict=cfg.get("max_new_tokens", 400), seed=cfg.get("seed"))
    return MapleChatModel(provider=provider or build_llm(cfg, use_adapter=use_adapter))


# ---- prompts (LCEL). Variables are filled at call time; evidence text is passed as a value, never parsed as template.
COMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    MessagesPlaceholder("history"),
    ("human", "{context}\n\nQUESTION: {question}"),
])
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question using Canadian clinical-guideline vocabulary for search. Output only the rewritten query."),
    ("human", "{question}"),
])
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Classify the request into one of: info_mental, info_lifestyle, navigation, screening, out_of_scope, ambiguous. "
               "Reply with the label only."),
    ("human", "{question}"),
])


def usage(msg: AIMessage) -> tuple[int, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    return u.get("input_tokens", 0), u.get("output_tokens", 0)
