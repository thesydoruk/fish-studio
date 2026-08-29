# Security

## Supported versions

Security fixes are accepted against the default branch (`main`).

## Reporting a vulnerability

Do not open a public issue for a vulnerability.

Use [GitHub private vulnerability reporting](https://github.com/thesydoruk/fish-studio/security/advisories/new)
on this repository. Include:

- a description of the issue
- steps to reproduce, or a proof of concept
- the impact you expect

Please give a reasonable window before any public disclosure.

## Scope notes

This project can clone a voice from a short reference clip and can download
public media for dataset building. Operators are responsible for:

- keeping `.env` and `HF_TOKEN` off the public internet
- binding the HTTP server only where they intend (`INFERENCE_HOST`)
- having rights to every reference clip and training source they use
