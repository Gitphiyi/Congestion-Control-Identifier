import socket

CAPTURE_IP = "127.0.0.1"
CAPTURE_PORT = 1026


# TCP send
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((CAPTURE_IP, CAPTURE_PORT))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"bbr") # SETS CONGESTION CONTROL ALGO
    s.sendall(b"Hello, world")
    data = s.recv(1024)

print(f"Received {data!r}")