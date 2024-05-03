def convert_and_intersect(dictionary):
    # 初始化一个空集合，用于存储第一个列表转化的集合
    intersect_set = None

    # 遍历字典中的每个键值对
    for key, double_list in dictionary.items():
        # 对每个双重列表中的列表转化为集合并计算交集
        current_set = set(double_list[0])
        for sublist in double_list[1:]:
            current_set.intersection_update(set(sublist))

        # 如果是第一次设置intersect_set，直接赋值，否则计算与前面的交集
        if intersect_set is None:
            intersect_set = current_set
        else:
            intersect_set.intersection_update(current_set)

    # 将最终的交集集合转换回双重列表形式
    # 这里使用list来转换集合为列表，并将结果封装在一个外层列表中
    result_double_list = [list(intersect_set)]

    return result_double_list


# 示例字典
dictionary = {
    'key1': [['A', 'B', 'C'], ['B', 'C', 'D']],
    'key2': [['B', 'C', 'D'], ['C', 'D', 'E']]
}

# 调用函数并打印结果
result = convert_and_intersect(dictionary)
print(result)