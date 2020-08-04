from collections import Counter


def counter_statistics(waiting):
    """
    统计List中每个元素的出现次数
    :param waiting: 待处理的list
    :return: 无返回
    """
    value = waiting.numpy()
    result = Counter(value.tolist())
    summary = sum(result.values())
    result = result.items()

    proportion = []

    for key, v in result:
        proportion.append((key, v / summary))
    print(sorted(result))
    print(sorted(proportion))
