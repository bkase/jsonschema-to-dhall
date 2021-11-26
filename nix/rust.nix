{ sources ? import ./sources.nix }:
let
  pkgs =
    import sources.nixpkgs { overlays = [ (import sources.nixpkgs-mozilla) ]; };
  channel = "nightly";
  date = "2021-11-25";
  targets = if pkgs.stdenv.isDarwin then [ "x86_64-apple-darwin" ] else [ "x86_64-unknown-linux-gnu" ];
  rust1 = pkgs.rustChannelOfTargets channel date targets;
  rustc = rust1.override {
    inherit targets;
    extensions = [ "rust-src" "rls-preview" "rust-std" "rust-analysis" "rustfmt-preview" "clippy-preview" ];
  };
  pkgs2 =
    import sources.nixpkgs {
      overlays =
        [
          (import sources.nixpkgs-mozilla)
          (
            self: super: {
              # Replace "latest.rustChannels.stable" with the version of the rust tools that
              # you would like. Look at the documentation of nixpkgs-mozilla for examples.
              #
              # NOTE: "rust" instead of "rustc" is not a typo: It will include more than needed
              # but also the much needed "rust-std".
              rustc = rustc;
              # inherit (rustc) cargo rust rust-fmt rust-std clippy;
            }
          )

        ];
    };
in
{
  rustc = rustc;
  pkgs = pkgs2;
}
