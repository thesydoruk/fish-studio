# Contributing

Thanks for considering a patch. This repo is the Fish Studio source: HTTP TTS
server, dataset builder, and LoRA training around Fish Speech s2-pro.

## Setup

```bash
./run.sh install dev
cp .env.example .env
pytest
```

`./run.sh install dev` installs extras, applies the protobuf override, and
enables the pre-commit hooks (Ruff + Prettier).

## Pull requests

- Keep the change focused. Do not mix refactors with a bug fix.
- Match the surrounding style. Ruff is the formatter (`line-length = 100`).
- Add or update tests under `tests/` when behavior changes.
- Do not commit `.env`, `data/`, checkpoints, or training runs.
- Do not commit secrets, Hugging Face tokens, or live YouTube source lists.

## License

Contributions are accepted under the MIT License (`LICENSE`), copyright
Valerii Sydoruk unless you state otherwise in the commit.

Fish Speech and s2-pro remain under the Fish Audio Research License. Do not
vendor those sources or weights into this tree. See `licenses/NOTICE`.
