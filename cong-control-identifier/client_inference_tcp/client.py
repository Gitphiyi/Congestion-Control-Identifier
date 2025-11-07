import socket, ssl

host = "www.linkedin.com"
endpoint = "/"
port = 443

raw = socket.create_connection((host, port))
context = ssl.create_default_context()

chunks = []

with context.wrap_socket(raw, server_hostname=host) as sock:
    # Send a simple HTTP/1.1 GET (plain TCP)
    req_str = "GET " + endpoint + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n"
    req = req_str.encode("utf-8")
    sock.sendall(req)

    # Read response
    while True:
        data = sock.recv(4096)
        if not data:
            break
        chunks.append(data)


response = b"".join(chunks)
print(response.decode(errors="replace"))
