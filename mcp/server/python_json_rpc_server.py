import sys
import json

def send_message(msg):
    # Serialize and send a JSON-RPC message followed by a newline
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle_request(req):
    method = req.get("method")
    id_ = req.get("id")

    # Handle the "initialize" handshake
    if method == "initialize":
        result = {"capabilities": {}}
        return {"jsonrpc": "2.0", "id": id_, "result": result}

    # Example custom method: "echo"
    elif method == "echo":
        params = req.get("params", {})
        return {"jsonrpc": "2.0", "id": id_, "result": params}

    # Method not found
    else:
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "error": {"code": -32601, "message": "Method not found"}
        }


def main():
    # Read lines from stdin, parse them as JSON-RPC requests
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            send_message(resp)
        except Exception as e:
            # On error, send a JSON-RPC internal error response
            error_msg = {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "error": {"code": -32603, "message": str(e)}
            }
            send_message(error_msg)

if __name__ == "__main__":
    main()