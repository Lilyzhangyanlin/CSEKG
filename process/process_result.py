""" 处理实验结果 @230221

"""


# 将这里修改为需要处理的结果字符
S = """
OrderedDict([('recall@20', 0.3548), ('mrr@20', 0.1201), ('ndcg@20', 0.1716), ('hit@20', 0.3589), ('precision@20', 0.0182)])
# cls
OrderedDict([('recall@20', 0.35), ('mrr@20', 0.1092), ('ndcg@20', 0.1617), ('hit@20', 0.3535), ('precision@20', 0.0178)])

"""

from collections import OrderedDict
keys = ['recall@20', 'ndcg@20', 'precision@20']
def get_result(s):
    """ 抽取如上格式的实验结果中的数据, 方便转到Excel中 """
    for line in s.split('\n'):
        if not 'OrderedDict' in line: continue
        d = eval(line.strip())
        # 这里可以调整小数点的数量
        result = [f"{d[k]:.4f}" for k in keys]
        # 按照 \t 进行打印, 可以用 , 进行分割存储为csv等
        # print("\t".join(result))
        print(",".join(result))

get_result(S)
