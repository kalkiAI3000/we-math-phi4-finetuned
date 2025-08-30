#!/usr/bin/env python3 \n\n\n
import os
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Fixed sample inputs
caption = "A honeycomb-like grid pattern made of connected hexagons."
question = (
    "As shown in the figure, which of the following shapes is the basic unit of a honeycomb? "
    "A. Parallelogram; B. Regular hexagon; C. Square; D. Regular pentagon"
)
image_path = "/data-mount-large/scripts/test.jpeg"

# HF model repo (finetuned)
MODEL_ID = "kalkiai3000/we-math-phi4"
# Use base processor to avoid audio feature extractor issues
PROCESSOR_ID = "microsoft/Phi-4-multimodal-instruct"


def detect_mcq(text: str) -> bool:
    if re.search(r"(^|\s)[A-D][\.:]\s", text):
        return True
    if all(x in text for x in ["A:", "B:", "C:"]):
        return True
    if all(x in text for x in ["A.", "B.", "C."]):
        return True
    if ";" in text and all(opt in text for opt in ["A", "B", "C"]):
        return True
    return False


def build_prompt(question_text: str, caption_text: str) -> str:
    is_mcq = detect_mcq(question_text)
    if is_mcq:
        instruction = "Answer with the option's letter from the given choices directly."
    else:
        instruction = "Answer succinctly with the final value/word only."
    return (
        f"<|user|><|image_1|>Please solve this math problem: {question_text}\n"
        f"Image description: {caption_text}\n{instruction}<|end|><|assistant|>"
    )


def extract_answer(text: str, is_mcq: bool) -> str:
    t = text.strip()
    t = re.sub(r"^\s*The answer is\s*:\s*", "", t, flags=re.IGNORECASE)
    if is_mcq:
        m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
        return m.group(1).upper() if m else t[:1].upper()
    tokens = re.findall(r"[A-Za-z0-9\.]+", t)
    return tokens[-1] if tokens else t


def main() -> None:
    os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")
    # Load processor and model
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > 1024:
        try:
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        except Exception:
            image = image.resize((1024, 1024))

    # Build prompt
    prompt = build_prompt(question, caption)
    is_mcq = detect_mcq(question)

    # Processor forward
    proc_out = processor(prompt, images=[image], return_tensors="pt")

    # Move to GPU
    device = next(model.parameters()).device
    inputs = {
        "input_ids": proc_out.input_ids.to(device),
        "attention_mask": (proc_out.input_ids != processor.tokenizer.pad_token_id).long().to(device),
        "input_image_embeds": proc_out.input_image_embeds.to(device),
        "image_attention_mask": proc_out.image_attention_mask.to(device),
        "image_sizes": proc_out.image_sizes.to(device),
        "input_mode": torch.tensor([1], dtype=torch.long, device=device),
    }

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=4 if is_mcq else 64,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_logits_to_keep=1,
            use_cache=False,
        )

    # Decode only the continuation
    input_len = inputs["input_ids"].size(1)
    out_text = processor.batch_decode(
        gen_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Extract final answer
    final_answer = extract_answer(out_text, is_mcq)
    print(final_answer)


if __name__ == "__main__":
    main()
