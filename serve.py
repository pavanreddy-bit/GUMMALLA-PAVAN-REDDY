import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

if sys.version_info[0] < 3:
    import SimpleHTTPServer
    import SocketServer

    class Handler(SimpleHTTPServer.SimpleHTTPRequestHandler):
        pass

    Handler.extensions_map.update({
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.json': 'application/json',
        '.svg': 'image/svg+xml'
    })
    SocketServer.TCPServer.allow_reuse_address = True
    httpd = SocketServer.TCPServer(('127.0.0.1', PORT), Handler)
    print('CardioXAI server running at http://127.0.0.1:%d' % PORT)
    sys.stdout.flush()
    httpd.serve_forever()
else:
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        pass

    Handler.extensions_map = Handler.extensions_map.copy()
    Handler.extensions_map.update({
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.json': 'application/json',
        '.svg': 'image/svg+xml'
    })

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
        print('CardioXAI server running at http://127.0.0.1:%d' % PORT)
        sys.stdout.flush()
        httpd.serve_forever()
