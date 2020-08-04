from collections import Counter


def counter_statistics(waiting):
    """
    统计List中每个元素的出现次数
    """
    value = waiting.numpy()
    result = Counter(value.tolist())
    summary = sum(result.values())
    result = result.items()

    accuracy = []

    for key, v in result:
        accuracy.append((key, v / summary))
    print(result)
    print(accuracy)
