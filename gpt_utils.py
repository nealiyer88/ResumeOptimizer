import time
import openai
import tiktoken

GPT_CONTEXT_LIMIT = 7900
MAX_RETRIES = 5

# === Token Estimation for GPT Inputs ===
def num_tokens(text: str | list | dict, model: str = "gpt-4") -> int:
    """
    Returns the number of tokens for a given string, list of strings, or dict.
    Used to estimate token usage before sending to GPT.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if isinstance(text, str):
        return len(encoding.encode(text))
    elif isinstance(text, list):
        return sum(num_tokens(t, model=model) for t in text)
    elif isinstance(text, dict):
        return sum(num_tokens(v, model=model) for v in text.values())
    else:
        raise TypeError("num_tokens() input must be str, list[str], or dict")


# === Safe GPT Wrapper with Token & Rate Limit Handling ===
def gpt_safe_call(prompt_fn, fallback_prompt_fn=None, run_fn=None, fallback_return=None):
    """
    Universal GPT wrapper that:
    - Applies token limit fallback
    - Retries on RateLimitError
    - Inserts delay to avoid token-per-minute (TPM) overload
    """

    def run_with_retry(prompt):
        delay = 1
        for attempt in range(MAX_RETRIES):
            try:
                result = run_fn(prompt)
                time.sleep(1.2)  # throttle to avoid 10k token-per-minute cap
                return result
            except openai.RateLimitError:
                print(f"🔁 RateLimitError: retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"❌ GPT call failed: {e}")
                break
        print("⚠️ Max retries exceeded — using fallback.")
        return fallback_return

    try:
        prompt = prompt_fn()
        if num_tokens(prompt) < GPT_CONTEXT_LIMIT:
            print(f"[🧠 Using full prompt] Token count = {num_tokens(prompt)}")
            return run_with_retry(prompt)

        elif fallback_prompt_fn:
            trimmed = fallback_prompt_fn()
            if num_tokens(trimmed) < GPT_CONTEXT_LIMIT:
                print(f"[✂️ Using trimmed prompt] Token count = {num_tokens(trimmed)}")
                return run_with_retry(trimmed)
            else:
                print("⚠️ Even trimmed prompt exceeds GPT token limit.")
    except Exception as e:
        print(f"[GPT Safe Call Error] {e}")

    print("⚠️ Falling back to original unenhanced text.")
    return fallback_return
