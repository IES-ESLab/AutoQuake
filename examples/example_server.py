# test_server.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

messages = []  # store received messages in memory

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        
        msgs_html = "".join(f"<pre>{json.dumps(m, indent=2)}</pre><hr>" for m in messages)
        html = f"""
        <html>
        <head><meta http-equiv="refresh" content="2"></head>
        <body>
            <h2>Received Messages ({len(messages)})</h2>
            {msgs_html or "<p>No messages yet.</p>"}
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        messages.append(data)

        print('\n' + '='*60)
        print(f'Received: {self.path}')
        print(json.dumps(data, indent=2))
        print('='*60)

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), RequestHandler)
    print('Test server running on http://localhost:8080')
    server.serve_forever()