import socket

### Hosts
CAPTURE_IP = "127.0.0.1"
CAPTURE_PORT = 1026

SERVER_IP = "127.0.0.1"
SERVER_PORT = 1024

# Parameters
avail_bandwidth = 90 # in Bytes
additional_delay = 40 # in ms
drop_rate = 20 # percentage

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((CAPTURE_IP, CAPTURE_PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by client at addr {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.connect((SERVER_IP, SERVER_PORT))
                server_socket.sendall(data)
                server_data = server_socket.recv(1024)
            print(f"Received {data!r}")
            conn.sendall(server_data)
