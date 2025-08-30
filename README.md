# We-Math Phi-4 (Finetuned)

This repository documents how to use the finetuned Phi-4 Multimodal model for image-based math reasoning and how to reproduce a short base-vs-finetuned comparison on GPU.

- Finetuned model (Hugging Face): `kalkiai3000/we-math-phi4`
- Base model: `microsoft/Phi-4-multimodal-instruct`
- Dataset (captions): `kalkiai3000/we-math-captions`

## Installation

Create a Python environment (GPU recommended) and install:

```bash
pip install -r requirements.txt
```

## Single-datapoint prediction (GPU)

The snippet below loads the base processor and the finetuned model from the Hub, builds the prompt with the caption, and generates a concise answer (A/B/C/D for MCQ or a final numeric/word for word problems).

```python
import os
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Sample inputs
caption = "A honeycomb-like grid pattern made of connected hexagons."
question = (
    "As shown in the figure, which of the following shapes is the basic unit of a honeycomb? "
    "A. Parallelogram; B. Regular hexagon; C. Square; D. Regular pentagon"
)
image_path = "/data-mount-large/scripts/test.jpeg"  # change to your image path

MODEL_ID = "kalkiai3000/we-math-phi4"
PROCESSOR_ID = "microsoft/Phi-4-multimodal-instruct"

os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")
processor = AutoProcessor.from_pretrained(PROCESSOR_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
try: model.config.use_cache = False
except Exception: pass
try: model.gradient_checkpointing_disable()
except Exception: pass

# MCQ detection for instruction
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

is_mcq = detect_mcq(question)
if is_mcq:
    instruction = "Answer with the option's letter from the given choices directly."
    max_new = 4
else:
    instruction = "Answer succinctly with the final value/word only."
    max_new = 64

prompt = (
    f"<|user|><|image_1|>Please solve this math problem: {question}\n"
    f"Image description: {caption}\n{instruction}<|end|><|assistant|>"
)

# Prepare image and inputs
image = Image.open(image_path).convert("RGB")
if max(image.size) > 1024:
    try: image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    except Exception: image = image.resize((1024, 1024))

proc = processor(prompt, images=[image], return_tensors="pt")
device = next(model.parameters()).device
inputs = {
    "input_ids": proc.input_ids.to(device),
    "attention_mask": (proc.input_ids != processor.tokenizer.pad_token_id).long().to(device),
    "input_image_embeds": proc.input_image_embeds.to(device),
    "image_attention_mask": proc.image_attention_mask.to(device),
    "image_sizes": proc.image_sizes.to(device),
    "input_mode": torch.tensor([1], dtype=torch.long, device=device),
}

with torch.no_grad():
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=0.0,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_logits_to_keep=1,
        use_cache=False,
    )

# Decode continuation only and normalize
in_len = inputs["input_ids"].shape[1]
out_text = processor.batch_decode(gen[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

if is_mcq:
    m = re.search(r"\b([ABCD])\b", out_text, flags=re.IGNORECASE)
    print((m.group(1).upper() if m else out_text[:1]).strip())
else:
    tokens = re.findall(r"[A-Za-z0-9\.]+", out_text)
    print((tokens[-1] if tokens else out_text).strip())
```

## Base vs Finetuned comparison (100 samples, GPU)

This script compares the base model (no captions) vs the finetuned model (with captions) on the first 100 captioned examples from We-Math `train.json`.

- Outputs: JSON (`/data-mount-large/scripts/comparison_results.json`) and a Markdown report (`/data-mount-large/scripts/comparison.md`).
- Uses greedy decoding (temperature 0.0) and normalizes answers (single-letter for MCQ; final token/value for word problems).

Run:

```bash
source activate ft-model && python /data-mount-large/scripts/compare_models.py
```

See the generated `comparison.md` for a summary and concrete win cases where the base fails and finetuned passes.

## Notes
- We recommend running on a GPU for both prediction and comparison.
- The base processor is used for both models; captions are only appended to the finetuned prompts.
