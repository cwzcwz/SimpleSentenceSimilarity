import pandas as pd
import tensorflow as tf


def load_dataset(data_str, mid_file_path, final_file_path):
    df = pd.read_json(data_str, orient='records')
    df.to_csv(mid_file_path, index=False, header=0)

    sent_pairs = []
    with tf.gfile.GFile(mid_file_path, "r") as f:
        for line in f:
            ts = line.strip().split(",")
            sent_pairs.append((int(ts[4]), int(ts[2]), ts[3], ts[0]))

    final_pairs = []
    # 结算列表的长度
    n = len(sent_pairs)
    # 外层循环控制从头走到尾的次数
    for j in range(0, n - 1):
        # 内层循环控制走一次的过程
        for i in range(1 + j, n - 1):
            # 如果前一个元素大于后一个元素，则交换两个元素（升序）
            if sent_pairs[j][0] == sent_pairs[i][0]:
                final_pairs.append((sent_pairs[j][0], sent_pairs[j][1], sent_pairs[i][1], sent_pairs[j][2],
                                    sent_pairs[i][2], sent_pairs[j][3], sent_pairs[i][3]))

    final = pd.DataFrame(final_pairs,
                         columns=["video_id", "id1", "id2", "question1", "question2", "answer1", "answer2"])

    ##用于保存最终的结果
    final.to_csv(final_file_path, index=False)

    return final


if __name__ == '__main__':
    data_str1 = "./data/test_qa.json"
    file_path = "./data/test_qa1.c  sv"
    final_file_path = "./data/final_test_qa1.csv"
    load_dataset(data_str1, file_path, final_file_path)
