# Json-schema to Dhall

Currently it is specialized to Buildkite's pipeline schema.

To build:

```bash
nix-build
make release
```

## Status

With (minor) tweaks to the input `schema.json` and a small patch-file to fix one issue in the codegen, the command step, and therefore all of it's dependencies, are working.

See `out/` directory for generated Dhall, `out/definitions/commandStep/Type` is the command step.

