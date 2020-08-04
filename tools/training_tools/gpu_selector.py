import torch.cuda as cuda


def gpu_selector():
    """
    查看GPU相关信息
    :return: gpu_available
    """
    gpu_available = cuda.is_available()
    device_name = cuda.get_device_name(1)
    device_capability = cuda.get_device_capability(1)
    print(gpu_available)
    print(device_name)
    print(device_capability)
    if gpu_available:
        print("device_number is ", 1)
        cuda.set_device(1)
    return gpu_available
