import argparse
import time
import os
import http.server
import socketserver
import subprocess
import cgi
import json
import urllib.parse

PORT = 8001
PYTHON_SCRIPT = "../../model/tomb_finder/tomb_model.py"
OUTPUT_DIR = "/home/natalie/Documents/GitHub/ChamberCrawlers/presentation/website/simulation_vids"

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == '/simulate':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers.get('Content-Type')},
                encoding='utf-8'
            )
            num_agents = form.getvalue('num_agents')
            elevation_weight = form.getvalue('elevation_weight')
            tomb_distance_weight = form.getvalue('tomb_distance_weight')
            agent_distance_weight = form.getvalue('agent_distance_weight')
            slope_weight = form.getvalue('slope_weight')
            print(f"Received data: num_agents={num_agents}, elevation_weight={elevation_weight}, tomb_distance_weight={tomb_distance_weight}, agent_distance_weight={agent_distance_weight}, slope_weight={slope_weight}")
            command = [
                "python",
                PYTHON_SCRIPT,
                "--num_agents", num_agents,
                "--elevation_weight", elevation_weight,
                "--tomb_distance_weight", tomb_distance_weight,
                "--agent_distance_weight", agent_distance_weight,
                "--slope_weight", slope_weight,
                "--output_dir", OUTPUT_DIR
            ]
            print(" ".join(command))
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                stdout_str = stdout.decode('utf-8')
                stderr_str = stderr.decode('utf-8')
                if process.returncode == 0:
                    print(f"Python script executed successfully. Output: {stdout_str}")
                    video_path = None
                    for line in stdout_str.splitlines():
                        if "Video path:" in line:
                            video_path = line.split("Video path:")[1].strip()
                            break
                    if video_path:
                        # 1. Load the video file
                        try:
                            video_file = open(video_path, 'rb')
                            video_data = video_file.read()
                            video_file.close()
                        except Exception as e:
                            error_message = f"Error reading video file: {e}"
                            print(error_message)
                            self.send_response(500)
                            self.send_cors_headers()
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))
                            return

                        # 2. Send the video data
                        self.send_response(200)
                        self.send_cors_headers()
                        self.send_header('Content-type', 'video/mp4')
                        self.send_header('Content-length', len(video_data))
                        self.end_headers()
                        self.wfile.write(video_data)
                    else:
                        error_message = "Python script did not return the video path."
                        print(error_message)
                        self.send_response(500)
                        self.send_cors_headers()
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))
                else:
                    error_message = f"Python script failed with code {process.returncode}. Error: {stderr_str}"
                    print(error_message)
                    self.send_response(500)
                    self.send_cors_headers()
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))
            except Exception as e:
                error_message = f"Error executing Python script: {e}"
                print(error_message)
                self.send_response(500)
                self.send_cors_headers()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))
        else:
            super().do_GET()

    def do_GET(self):
        self.send_cors_headers()
        super().do_GET()

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()