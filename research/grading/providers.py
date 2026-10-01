"""Resolve hosted model specifications with credential checks and bounded retries."""

from __future__ import annotations

import os

from httpx2 import AsyncClient, HTTPStatusError, TransportError
from pydantic_ai.models import Model

try:
    from pydantic_ai.retries import (
        AsyncHTTPX2TenacityTransport,
        RetryConfig,
        wait_retry_after,
    )
except ImportError:
    from pydantic_ai.retries import (
        AsyncTenacityTransport as AsyncHTTPX2TenacityTransport,
    )
    from pydantic_ai.retries import RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

# Respect server backoff while bounding each transport retry cycle.
MAX_ATTEMPTS = 10
MAX_WAIT_SECONDS = 300
# Short generation timeouts would systematically drop harder, longer-reasoning items.
REQUEST_TIMEOUT = float(os.environ.get("LOCALGATE_REQUEST_TIMEOUT", "600"))

# Providers that speak OpenAI's API but are not built in. Everything else is addressed by
# Pydantic AI's own `provider:model` shorthand, so this stays short by design.
OPENAI_COMPATIBLE: dict[str, tuple[str, str]] = {
    "gc": ("https://api.generalcompute.com/v1", "GENERALCOMPUTE_API_KEY"),
    "merge": ("https://api-gateway.merge.dev/v1/openai", "MERGE_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "local": (os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"), "LOCAL_API_KEY"),
}

# Which environment variable holds each built-in provider's key, so a missing one can be
# named before any request is made.
BUILTIN_KEY_ENV: dict[str, str] = {
    # Bedrock accepts this bearer token or credentials from the AWS credential chain.
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "CO_API_KEY",
}


class ProviderError(RuntimeError):
    """A spec that cannot be resolved, or a key that is not set."""


def retrying_client() -> AsyncClient:
    """Create an HTTP client with bounded retries and Retry-After handling."""
    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            # Connection failures and read timeouts never reach a status code.
            retry=retry_if_exception_type((HTTPStatusError, TransportError)),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=MAX_WAIT_SECONDS,
            ),
            stop=stop_after_attempt(MAX_ATTEMPTS),
            reraise=True,
        ),
        # A status is only retryable once it has been raised, so raise it here.
        validate_response=lambda response: response.raise_for_status(),
    )
    return AsyncClient(transport=transport, timeout=REQUEST_TIMEOUT)


def key_env(spec: str) -> str:
    """Name the environment variable a spec's key comes from."""
    provider, _, model = spec.partition(":")
    if not model:
        raise ProviderError(f"{spec!r} is not 'provider:model'")
    if provider in OPENAI_COMPATIBLE:
        return OPENAI_COMPATIBLE[provider][1]
    if provider in BUILTIN_KEY_ENV:
        return BUILTIN_KEY_ENV[provider]
    raise ProviderError(
        f"unknown provider {provider!r}. Built in: {', '.join(sorted(BUILTIN_KEY_ENV))}; "
        f"OpenAI-compatible: {', '.join(sorted(OPENAI_COMPATIBLE))}"
    )


def require_keys(*specs: str) -> None:
    """Check provider credentials before making requests, reporting variable names only."""
    missing = set()
    for spec in specs:
        env = key_env(spec)
        provider, _, _ = spec.partition(":")
        if provider == "bedrock":
            if not os.environ.get(env):
                import boto3
                if boto3.session.Session().get_credentials() is None:
                    missing.add(env)
        elif not os.environ.get(env):
            missing.add(env)
    if missing:
        raise ProviderError(f"missing API key(s): {', '.join(sorted(missing))}")


def resolve(spec: str) -> Model | str:
    """Resolve a model with provider-specific authentication and retries.

    OpenAI-compatible and DeepSeek requests use the bounded HTTP retry client.
    Bedrock uses adaptive SDK retries. Other built-in providers use Pydantic AI's
    provider defaults.
    """
    require_keys(spec)
    provider_name, _, model_name = spec.partition(":")

    if provider_name in OPENAI_COMPATIBLE:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base_url, env = OPENAI_COMPATIBLE[provider_name]
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=base_url, api_key=os.environ[env], http_client=retrying_client()
            ),
        )

    if provider_name == "bedrock":
        # Bedrock uses the AWS SDK, so its retries are configured separately from HTTP.
        # Adaptive mode handles throttling. An explicit bearer token takes precedence
        # over the standard credential chain when supplied.
        # Each output-validation attempt can make up to MAX_ATTEMPTS transport calls;
        # the batch failure limit counts failed items after these retries finish.
        import boto3
        from botocore.config import Config
        from botocore.exceptions import NoRegionError
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        try:
            # A fresh session, not boto3's module-level default: the default
            # session caches environment resolution from its first use, which
            # makes region/credential changes between calls invisible.
            config_kwargs: dict = {
                "retries": {"total_max_attempts": MAX_ATTEMPTS, "mode": "adaptive"},
                "read_timeout": REQUEST_TIMEOUT,
                "connect_timeout": 60,
            }
            if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
                config_kwargs["signature_version"] = "bearer"
            client = boto3.session.Session().client(
                "bedrock-runtime",
                config=Config(**config_kwargs),
            )
        except NoRegionError as err:
            raise ProviderError(
                "no AWS region configured — set AWS_DEFAULT_REGION (this botocore "
                "ignores AWS_REGION) or configure ~/.aws before a Bedrock batch"
            ) from err
        return BedrockConverseModel(model_name, provider=BedrockProvider(bedrock_client=client))

    if provider_name == "deepseek":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        return OpenAIChatModel(
            model_name,
            provider=DeepSeekProvider(
                api_key=os.environ[BUILTIN_KEY_ENV[provider_name]], http_client=retrying_client()
            ),
        )

    return spec
