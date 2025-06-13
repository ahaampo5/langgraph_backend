
# File: stdio_client.py
# ```python
# #!/usr/bin/env python3
import subprocess
import json

def send_request(proc, msg):
    # Write a JSON-RPC request to the server's stdin
    serialized = json.dumps(msg) + "\n"
    proc.stdin.write(serialized.encode('utf-8'))
    proc.stdin.flush()

    # Read a line from stdout and parse it as JSON
    response_line = proc.stdout.readline().decode('utf-8').strip()
    print("Received response:", response_line)
    return json.loads(response_line)


def main():
    # Launch the stdio-based MCP server subprocess
    proc = subprocess.Popen(
        ["python3", "../server/python_json_rpc_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 1) Initialize handshake
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = send_request(proc, init_req)
    print("Initialize response:", init_resp)

    # 2) Send an "echo" request
    echo_req = {"jsonrpc": "2.0", "id": 2, "method": "echo", "params": {"message": "Hello MCP"}}
    echo_resp = send_request(proc, echo_req)
    print("Echo response:", echo_resp)

    # Terminate the server subprocess cleanly
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()