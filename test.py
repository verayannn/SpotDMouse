import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")

    # You can also get information about the current device
    current_device_index = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device_index)
    print(f"Current active GPU: {current_device_name} (index: {current_device_index})")

else:
    print("CUDA is not available. PyTorch will use the CPU.")
