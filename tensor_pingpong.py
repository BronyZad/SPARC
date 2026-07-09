import torch
import zmq
import time
import io
import argparse

# Simulated KV Cache Sizes for Qwen-4B at 4K Context
NATIVE_MB = 320
SABER_MB = 102

def create_dummy_tensor(size_in_mb):
    """Creates a random BF16 tensor of roughly the target megabyte size."""
    # 1 MB of bf16 is 524,288 elements
    num_elements = size_in_mb * 524288
    return torch.randn(num_elements, dtype=torch.bfloat16)

def start_receiver(port="5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"🎧 Receiver listening on port {port}...")

    while True:
        # Wait for the next request from client
        message = socket.recv()
        
        # Deserialize the PyTorch tensor
        buffer = io.BytesIO(message)
        tensor = torch.load(buffer, map_location="cpu")
        
        size_mb = len(message) / (1024 * 1024)
        print(f"✅ Received Tensor: {size_mb:.1f} MB")

        # Send back a quick acknowledgment
        socket.send_string("ACK")

def start_sender(ip, port="5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print(f"🚀 Connecting to receiver at {ip}:{port}...")
    socket.connect(f"tcp://{ip}:{port}")

    payloads = [
        ("Native-Baseline", NATIVE_MB),
        ("Sparc-Compressed", SABER_MB)
    ]

    for name, size_mb in payloads:
        print(f"\n========================================")
        print(f" 📦 PREPARING: {name} (~{size_mb} MB)")
        print(f"========================================")
        
        # 1. Generate Dummy Data
        tensor = create_dummy_tensor(size_mb)
        
        # 2. Serialize (Time this)
        t0_ser = time.perf_counter()
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        byte_data = buffer.getvalue()
        ser_time = (time.perf_counter() - t0_ser) * 1000
        
        actual_mb = len(byte_data) / (1024 * 1024)
        print(f"Serialization Time: {ser_time:.1f} ms | Payload Size: {actual_mb:.1f} MB")

        # 3. Transmit (Time this)
        t0_net = time.perf_counter()
        socket.send(byte_data)
        
        # Wait for acknowledgment
        socket.recv()
        net_time = (time.perf_counter() - t0_net) * 1000
        
        print(f"Transmission Time:  {net_time:.1f} ms ({(net_time/1000):.2f} seconds)")
        time.sleep(2) # Give the network a breath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['sender', 'receiver'], help="Run as sender or receiver")
    parser.add_argument('--ip', type=str, default="127.0.0.1", help="IP of the receiver (used by sender)")
    args = parser.parse_args()

    if args.mode == "receiver":
        start_receiver()
    else:
        start_sender(args.ip)
