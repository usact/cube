import requests
from transformers.utils import logging as hf_logging
from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError

def safe_from_pretrained(load_fn, repo_id: str, *args, **kwargs):
    """
    Try load_fn(repo_id, local_files_only=True, ...).
    If not found in cache, try load_fn(repo_id, local_files_only=False, ...).
    If that fails due to network, raise immediate error.
    
    load_fn: classmethod like AutoencoderKL.from_pretrained or
             CLIPTokenizer.from_pretrained
    repo_id: HuggingFace repo identifier
    *args/**kwargs: passed to load_fn
    """
    prev_verb = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # suppress HF logging during offline check
    
    # 1) Offline-only
    try:
        print(f"🔍 Checking cache for {repo_id} ... ")
        return load_fn(repo_id, local_files_only=True, *args, **kwargs)
    except (LocalEntryNotFoundError, RepositoryNotFoundError, OSError, AttributeError, ValueError) as e:
        # not in cache; restore verbosity and fall back
        hf_logging.set_verbosity(prev_verb)
        print(f"⚡ {repo_id} not in cache due to error {e} → attempting download…")
        try:
            return load_fn(repo_id, local_files_only=False, *args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"❌ Unable to download `{repo_id}` (no internet?).\n"
                "   • To pre-cache, run:\n"
                f"       transformers-cli download {repo_id}\n"
                "   • Or ensure HF cache is populated at `~/.cache/huggingface/`\n"
            ) from e
        finally:
            hf_logging.set_verbosity(prev_verb)
    finally:
        hf_logging.set_verbosity(prev_verb)
