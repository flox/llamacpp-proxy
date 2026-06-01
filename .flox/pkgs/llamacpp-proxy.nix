{ rustPlatform, lib }:
let
  root = ../../.;
in
rustPlatform.buildRustPackage {
  pname = "llamacpp-proxy";
  version = "0.1.0";
  src = root;
  cargoLock.lockFile = "${root}/Cargo.lock";
  meta = {
    description = "Unified API translation proxy for llama-server coding-agent harness compatibility";
    license = with lib.licenses; [ mit asl20 ];
  };
}
