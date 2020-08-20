def voting(predicts, file_name):
    """
    对一整段语音进行投票，分为n个3s的数据
    :param predicts: 神经网络每一类别的预测概率
    :param file_name: 文件名称
    :returns results: 给出的预测结果
    """
    result_dictionary = {}
    count_dictionary = {}

    for file, predict in zip(file_name, predicts):
        if file not in result_dictionary.keys():
            result_dictionary[file] = [0, 0, 0, 0, 0, 0]
            count_dictionary[file] = 0

        # 统计数目
        result_dictionary[file][predict] += 1

        # if max(predict) >= 0.9:
        # count_dictionary[file] += 1
        # result_dictionary[file][predict.argmax()] += 1
        # result_dictionary[file] = [result_dictionary[file][i] + predict[i] for i in range(len(result_dictionary[file]))]

        # # 进行投票选举，剔除掉废票
        # if max(predict) >= 0.9:
        #     result_dictionary[file] = [result_dictionary[file][i] + predict[i] for i in range(len(result_dictionary[file]))]
    results = {}
    for key, value in result_dictionary.items():
        # if max(value) == count_dictionary[key]:
        results[key] = value.index(max(value))

        # value = value.data.numpy().max()
        # results[key] = int(round(value / count_dictionary[key]))

    # return list(file_name), list(predicts)
    # return list(results.keys()), list(results.values())
    return results


def final_voting(results):
    """
    对不同模型的结果进行投票，分为n个3s的数据
    :param results: 神经网络每一类别的预测概率
    :param files: 文件名称
    :returns results: 给出的预测结果
    """
    result_dictionary = {}

    for result in results:
        # 此时得到的result为一个字典
        for key, value in result.items():
            if key not in result_dictionary.keys():
                result_dictionary[key] = [0, 0, 0, 0, 0, 0]

            result_dictionary[key][value] += 1

    final_results = {}
    for key, value in result_dictionary.items():
        final_results[key] = value.index(max(value))

    return list(final_results.keys()), list(final_results.values())