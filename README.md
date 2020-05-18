# Json-schema to Dhall

Currently it is specialized to Buildkite's pipeline schema.

To build:

```bash
nix-build
make release
```

## Status

With (minor) tweaks to the input `schema.json` and a small patch-file to fix one issue in the codegen, the command/trigger/nestedCommand steps, and all of their dependencies, are working.

See `out/` directory for generated Dhall, everything referenced in `out/top_level/Type` is working.

Reference the "raw" URL for `out/top_level/Type` to use in your Dhall code.

