# models/ - Local Model Storage

This directory stores locally downloaded models for faster loading.

## Downloaded Models

After running `python scripts/download_vietnamese_model.py`, you'll have:

```
models/
└── vietnamese-embedding/
    ├── config.json
    ├── config_sentence_transformers.json
    ├── modules.json
    ├── pytorch_model.bin
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

## Usage

Set environment variable to use local model:

```bash
export VIETNAMESE_MODEL_PATH="./models/vietnamese-embedding"
```

Or in `.env`:
```
VIETNAMESE_MODEL_PATH=./models/vietnamese-embedding
```

## Benefits

- ✅ No download wait time
- ✅ Faster model loading
- ✅ Works offline
- ✅ Consistent model version

## Size

- Vietnamese embedding model: ~280 MB
