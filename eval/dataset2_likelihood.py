"""Optional teacher-forced continuation scoring, with bounded sequence chunks."""


def continuation_logprobs(
    model, tokenizer, prompt, continuation, include_eos=False, chunk_tokens=64
):
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    context = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    full = tokenizer(rendered + continuation, add_special_tokens=False)["input_ids"]
    if full[: len(context)] != context:
        raise ValueError(
            "Tokenizer merges across assistant boundary; explicit token alignment required"
        )
    target = full[len(context) :]
    if include_eos:
        eos = model.generation_config.eos_token_id
        if isinstance(eos, list):
            raise ValueError("Choose an explicit EOS policy for multiple stop tokens")
        target = target + [eos]
    return token_sequence_logprobs(model, context, target, chunk_tokens)


def token_sequence_logprobs(model, context, target, chunk_tokens=64):
    import torch

    if not context or not target:
        raise ValueError("Nonempty context and continuation required")
    full = context + target
    if len(full) > model.config.max_position_embeddings:
        raise ValueError("Scoring sequence exceeds context")
    logps = []
    # Prefix recomputation trades compute for bounded output vocabulary memory.
    # logits_to_keep limits the language-model head to the chunk being scored.
    with torch.inference_mode():
        for start in range(0, len(target), chunk_tokens):
            end = min(start + chunk_tokens, len(target))
            prefix = full[: len(context) + end - 1]
            ids = torch.tensor([prefix], device=model.device)
            logits = (
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=end - start,
                )
                .logits[0]
                .float()
            )
            labels = torch.tensor(target[start:end], device=model.device)
            selected = logits.gather(-1, labels[:, None]).squeeze(-1) - torch.logsumexp(
                logits, dim=-1
            )
            logps.extend(selected.cpu().tolist())
    return {
        "token_logprobs": logps,
        "sum_logprob": sum(logps),
        "mean_logprob": sum(logps) / len(logps),
        "token_count": len(logps),
        "continuation_ids": target,
    }


def likelihood_margin(model, tokenizer, prompt, refusal, compliance, **kwargs):
    r = continuation_logprobs(model, tokenizer, prompt, refusal, **kwargs)
    c = continuation_logprobs(model, tokenizer, prompt, compliance, **kwargs)
    return {
        "refusal": r,
        "compliance": c,
        "margin": r["sum_logprob"] - c["sum_logprob"],
    }
