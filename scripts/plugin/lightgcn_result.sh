
# 问题: 只使用了KG的一丢丢信息, 但是效果提升挺大的

# ml-1m
# base
18 May 10:37    INFO  best valid : OrderedDict([('recall@20', 0.1875), ('mrr@20', 0.1761), ('ndcg@20', 0.1183), ('hit@20', 0.5226), ('precision@20', 0.0467)])
18 May 10:37    INFO  test result: OrderedDict([('recall@20', 0.1936), ('mrr@20', 0.1944), ('ndcg@20', 0.1254), ('hit@20', 0.529), ('precision@20', 0.0498)])
# imp
18 May 10:18    INFO  best valid : OrderedDict([('recall@20', 0.3486), ('mrr@20', 0.3031), ('ndcg@20', 0.2305), ('hit@20', 0.7292), ('precision@20', 0.0802)])
18 May 10:18    INFO  test result: OrderedDict([('recall@20', 0.3658), ('mrr@20', 0.3454), ('ndcg@20', 0.2533), ('hit@20', 0.7377), ('precision@20', 0.0876)])
# cls
20 May 03:26    INFO  best valid : OrderedDict([('recall@20', 0.3197), ('mrr@20', 0.2913), ('ndcg@20', 0.2154), ('hit@20', 0.696), ('precision@20', 0.0769)])
20 May 03:26    INFO  test result: OrderedDict([('recall@20', 0.3292), ('mrr@20', 0.3249), ('ndcg@20', 0.2303), ('hit@20', 0.7008), ('precision@20', 0.0822)])

# lfm-small
# base
18 May 11:14    INFO  best valid : OrderedDict([('recall@20', 0.2656), ('mrr@20', 0.0843), ('ndcg@20', 0.1235), ('hit@20', 0.2686), ('precision@20', 0.0135)])
18 May 11:14    INFO  test result: OrderedDict([('recall@20', 0.2667), ('mrr@20', 0.0868), ('ndcg@20', 0.1253), ('hit@20', 0.27), ('precision@20', 0.0136)])
# imp
18 May 11:13    INFO  best valid : OrderedDict([('recall@20', 0.3436), ('mrr@20', 0.1138), ('ndcg@20', 0.1632), ('hit@20', 0.3488), ('precision@20', 0.0176)])
18 May 11:13    INFO  test result: OrderedDict([('recall@20', 0.3607), ('mrr@20', 0.12), ('ndcg@20', 0.1714), ('hit@20', 0.3664), ('precision@20', 0.0185)])
# cls
20 May 02:47    INFO  best valid : OrderedDict([('recall@20', 0.3458), ('mrr@20', 0.1129), ('ndcg@20', 0.1637), ('hit@20', 0.3509), ('precision@20', 0.0177)])
20 May 02:47    INFO  test result: OrderedDict([('recall@20', 0.354), ('mrr@20', 0.1221), ('ndcg@20', 0.1717), ('hit@20', 0.3583), ('precision@20', 0.0181)])

# amazon-book2
# # base
# OrderedDict([('recall@20', 0.2667), ('mrr@20', 0.0868), ('ndcg@20', 0.1253), ('hit@20', 0.27), ('precision@20', 0.0136)])
# # imp
# OrderedDict([('recall@20', 0.2051), ('mrr@20', 0.0803), ('ndcg@20', 0.1035), ('hit@20', 0.2318), ('precision@20', 0.0126)])
# # cls
# OrderedDict([('recall@20', 0.1685), ('mrr@20', 0.0626), ('ndcg@20', 0.0821), ('hit@20', 0.1923), ('precision@20', 0.0103)])

# ml-1m
# base
OrderedDict([('recall@20', 0.1936), ('mrr@20', 0.1944), ('ndcg@20', 0.1254), ('hit@20', 0.529), ('precision@20', 0.0498)])
# imp
OrderedDict([('recall@20', 0.3658), ('mrr@20', 0.3454), ('ndcg@20', 0.2533), ('hit@20', 0.7377), ('precision@20', 0.0876)])
# cls
OrderedDict([('recall@20', 0.3292), ('mrr@20', 0.3249), ('ndcg@20', 0.2303), ('hit@20', 0.7008), ('precision@20', 0.0822)])

# lfm-small
# base
OrderedDict([('recall@20', 0.2667), ('mrr@20', 0.0868), ('ndcg@20', 0.1253), ('hit@20', 0.27), ('precision@20', 0.0136)])
# imp
OrderedDict([('recall@20', 0.3482), ('mrr@20', 0.1152), ('ndcg@20', 0.1656), ('hit@20', 0.3535), ('precision@20', 0.0178)])
# cls
OrderedDict([('recall@20', 0.3417), ('mrr@20', 0.1159), ('ndcg@20', 0.1646), ('hit@20', 0.3439), ('precision@20', 0.0174)])