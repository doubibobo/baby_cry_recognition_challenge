import torch.cuda as cuda


def judge_and_choose_gpu():
    gpu_available = cuda.is_available()
    # 查看当前使用GPU的序号
    device_number = cuda.current_device()
    device_name = cuda.get_device_name(device_number)
    device_capability = cuda.get_device_capability(device_number)
    if gpu_available:
        print("device_number is ", device_number)
        cuda.set_device(0)
    return gpu_available, device_name, device_capability
