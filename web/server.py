from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import chess
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.nn as nn
import src.handmade as hn

TYPE = 'hn'
if len(sys.argv) > 1: TYPE = sys.argv[1]

class ChessRequestHandler(BaseHTTPRequestHandler):
    # Set a longer timeout for requests
    timeout = 300  # 5 minutes in seconds
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()
        
    def do_GET(self):
        # Serve the HTML file
        if self.path == '/' or self.path == '/index.html':
            try:
                # Get the path to index.html
                html_path = os.path.join(os.path.dirname(__file__), 'index.html')
                
                with open(html_path, 'r') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, "File not found")
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        if self.path == '/make-move':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            fen = data.get('fen', '')
            if not fen:
                self.send_response(400)
                self.end_headers()
                return
                
            try:
                new_fen = self.make_bot_move(fen)
                response = json.dumps({'fen': new_fen}).encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response)
            except Exception as e:
                print(f"Error in make_bot_move: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def make_bot_move(self, fen):
        board = chess.Board(fen)
        
        if board.is_game_over():
            return fen
        
        if TYPE == 'nn':
            root = nn.buildTree(board, 1)
            best_move = random.choice(root.children)
            board = chess.Board(best_move.board)
            return board.fen()
        else: hn.computer_move(board)
        
        return board.fen()

# Custom HTTP server with longer timeout
class TimeoutHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        # Set socket timeout to 5 minutes
        self.socket.settimeout(300)

if __name__ == '__main__':
    # Use our custom server with longer timeout
    server = TimeoutHTTPServer(('localhost', 8000), ChessRequestHandler)
    print('Starting server on http://localhost:8000')
    print('Open this URL in your browser to play chess')
    print('Server timeout set to 5 minutes')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down server')
        server.shutdown()
