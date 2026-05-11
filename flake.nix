{
  description = "swavelet — spiking wavelets in JAX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];

          shellHook = ''
            export PYTHONNOUSERSITE=1
            export UV_PYTHON=${python}/bin/python
            if [ ! -d .venv ]; then
              uv venv --python "$UV_PYTHON"
              uv pip install -e '.[test]'
            fi
            source .venv/bin/activate
          '';
        };
      });
}
