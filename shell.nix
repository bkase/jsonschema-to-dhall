let
  sources = import ./nix/sources.nix;
  rust = import ./nix/rust.nix { inherit sources; };
  pkgs = rust.pkgs;
in
pkgs.mkShell {
  name = "jsonschema-to-dhall";
  buildInputs = [
    pkgs.dhall
    rust.rustc
  ];
}
