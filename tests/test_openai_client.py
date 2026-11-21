import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from graphharness.models import OpenAIModel


def test_openai_client_posts_chat_completion():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps({
                "model": "test-model",
                "choices": [{"message": {"content": "hello from server"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = OpenAIModel(
            "test-model", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="secret", max_tokens=77,
        )
        result = model.chat([{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result.text == "hello from server"
    assert result.total_tokens == 5
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["max_tokens"] == 77
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
