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
            # Binary wheels (jaxlib) need a standard C++/zlib runtime on the path.
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            if [ ! -d .venv ]; then
              uv venv --python ${python}/bin/python
            fi
            source .venv/bin/activate
            uv pip install -e '.[test]'
          '';
        };
      });
}
