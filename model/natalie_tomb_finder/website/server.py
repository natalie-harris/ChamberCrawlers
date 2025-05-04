import http.server
import socketserver
import subprocess
import cgi
import os
import json

PORT = 8000  # You can change this port if needed
PYTHON_SCRIPT = "../tomb_model.py"  # Replace with the actual name of your Python script
OUTPUT_DIR = "../output" # make sure this directory exists

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/simulate':
            # 1. Read the form data from the POST request
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers.get('Content-Type')},
                encoding='utf-8'
            )

            # 2. Extract the slider values from the form data
            num_agents = form.getvalue('num_agents')
            elevation_weight = form.getvalue('elevation_weight')
            tomb_distance_weight = form.getvalue('tomb_distance_weight')
            agent_distance_weight = form.getvalue('agent_distance_weight')

            # 3. Print the data to the console (for debugging)
            print(f"Received data: num_agents={num_agents}, elevation_weight={elevation_weight}, tomb_distance_weight={tomb_distance_weight}, agent_distance_weight={agent_distance_weight}")

            # 4. Construct the command to execute your Python script
            #    Pass the slider values as command-line arguments
            command = [
                "python",
                PYTHON_SCRIPT,
                "--num_agents", num_agents,
                "--elevation_weight", elevation_weight,
                "--tomb_distance_weight", tomb_distance_weight,
                "--agent_distance_weight", agent_distance_weight,
                "--output_dir", OUTPUT_DIR # Pass the output directory
            ]

            # 5. Execute the Python script
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()  # Wait for the script to finish
                # Decode the output
                stdout_str = stdout.decode('utf-8')
                stderr_str = stderr.decode('utf-8')

                if process.returncode == 0:
                    print(f"Python script executed successfully. Output: {stdout_str}")
                    # 6.  (CRITICAL) Extract the video file path.  The python script MUST print the path
                    # The python script should print the video path to standard output.
                    #  For example:  "Video path: /path/to/your/video.mp4"
                    video_path = None
                    for line in stdout_str.splitlines():
                        if "Video path:" in line:
                            video_path = line.split("Video path:")[1].strip()
                            break

                    if video_path:
                        # 7. Send the video path back to the client as JSON
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'video_path': video_path}).encode('utf-8'))
                    else:
                        error_message = "Python script did not return the video path."
                        print(error_message)
                        self.send_response(500)  # Internal Server Error
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))

                else:
                    # 8. Handle errors from the Python script
                    error_message = f"Python script failed with code {process.returncode}. Error: {stderr_str}"
                    print(error_message)
                    self.send_response(500)  # Internal Server Error
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))

            except Exception as e:
                # 9. Handle exceptions during the process
                error_message = f"Error executing Python script: {e}"
                print(error_message)
                self.send_response(500)  # Internal Server Error
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': error_message}).encode('utf-8'))
        else:
            super().do_GET()  # Serve other files normally

# 10. Start the server
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()