"""
This is used by participants to serve their policy. 

Participants may need to modify this file to adapt to their own policy. for example, loading multiple model ckpts.

Provide an example usage here if using docker submission:

... (this can be mulitiple commands if you need to run multiple models in parallels)

"""

from challenge_interface.server import PolicyServer
from challenge_interface.server_http import PolicyHTTPServer
from challenge_interface.policy import DummyPolicy, SB3Policy
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a policy for the CVPR challenge.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=("websocket", "http"),
        default="websocket",
        help="Server transport to use (default: %(default)s).",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--policy",
        choices=("dummy", "sb3"),
        default="dummy",
        help="Which policy implementation to serve (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the SB3 checkpoint .zip (required when --policy=sb3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the SB3 policy (default: %(default)s).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.policy == "sb3":
        if not args.model:
            raise SystemExit("--policy sb3 requires --model <path-to-ckpt.zip>")
        policy = SB3Policy(model_path=args.model, device=args.device)
    else:
        policy = DummyPolicy()
    if args.transport == "http":
        server = PolicyHTTPServer(policy, host=args.host, port=args.port)
    else:
        server = PolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()

if __name__ == "__main__":
    main()