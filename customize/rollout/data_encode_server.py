import argparse
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI
from swift import get_processor, get_template

logger = logging.getLogger(__name__)

app = FastAPI()

template = None


@app.post("/encode_data")
async def encode_data(data: dict[str, Any]):
    encoded = template.encode(data)
    return {"input_ids": encoded["input_ids"]}


def parse_args():
    parser = argparse.ArgumentParser(description="Data encode server")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if "3.5" in args.tokenizer_path:
        model_type = "qwen3_5"
        template_type = "qwen3_5"
    else:
        model_type = "qwen3"
        template_type = "qwen3_thinking"
    tokenizer = get_processor(args.tokenizer_path, model_type=model_type)
    template = get_template(tokenizer, template_type=template_type)
    template.set_mode("train")
    logger.info("Loaded tokenizer from %s, agent_template: %s", args.tokenizer_path, template._agent_template)
    uvicorn.run(app, host=args.host, port=args.port)
