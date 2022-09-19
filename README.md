# 一、命令

进入文件夹：

```shell script
cd /home/ccq/Research/PLAN-reimp/Main
```

运行程序：

```shell script
nohup python main.py --gpu 0 &
```

# 二、数据集

| | max post num | avg post num | max word num | avg word num |
| :----: | :----: | :----: | :----: | :----: |
| Weibo | 599 | 168.81 | 280 | 17.47 |

# 三、实验

1. Weibo

- 实验结果

| id | dataset | runs | hitplan | max_length | max_tweet | emb_dim | d_model | d_feed_forward | vary_lr | lr | batch size | test acc | max acc |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | Weibo | 6 | False | no limit | **600** | 200 | 200 | 400 | True | - | **4** | 0.915±0.007 | 0.927 |
| 2 | Weibo | 6 | False | no limit | **400** | 200 | 200 | 400 | True | - | **8** | 0.906±0.007 | 0.915 |

2. Weibo-2class

- 实验结果

| id | dataset | runs | hitplan | max_length | max_tweet | emb_dim | d_model | d_feed_forward | vary_lr | lr | batch size | test acc | max acc |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | Weibo | 4 | False | no limit | **600** | 200 | 200 | 400 | True | - | **4** | 0.788±0.005 | 0.791 |
| 2 | Weibo | 1 | False | no limit | **400** | 200 | 200 | 400 | True | - | **8** | 0.786±0.000 | 0.786 |