let
  sources = import ./nix/sources.nix;
  rust = import ./nix/rust.nix { inherit sources; };
  pkgs = rust.pkgs;
in
let
  cargoNix = pkgs.callPackage ./Cargo.nix {
    pkgs = pkgs;
    defaultCrateOverrides = pkgs.defaultCrateOverrides // {
      jsonschema-to-dhall = attrs: {
        nativeBuildInputs = with pkgs;
          [ pkg-config ];
        buildInputs = with pkgs; [
          clang
        ] ++ (if pkgs.stdenv.isDarwin then [ pkgs.darwin.apple_sdk.frameworks.Security ] else []);
      };
    };
  };
in
let
  crate =
    cargoNix.rootCrate.build.override {
      runTests = true;
      testInputs = [] ++ (if pkgs.stdenv.isDarwin then [ pkgs.darwin.apple_sdk.frameworks.Security ] else []);
    };
in
pkgs.stdenv.mkDerivation {
  name = "dhall-bindings";
  src = ./.;
  # TODO: Build the rust package with nix instead of relying on cargo
  buildInputs = [ pkgs.patch pkgs.dhall pkgs.gnumake crate ];
  buildPhase = ''
    #!${pkgs.runtimeShell}
    make BIN=${crate}/bin/jsonschema-to-dhall generate
    cp -r out/ $out/
  '';
}
