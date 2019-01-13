# coding: utf-8

'''
    训练词向量
    python train_word2vec.py --src_data=src_file_path --model_path=model_file_path --vector_path=vector_file_path
'''

import logging
import os
import sys
import multiprocessing
import argparse         # 输入参数的包

import jieba
import jieba.analyse
import jieba.posseg as pseg

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



## hyperparameters
parser = argparse.ArgumentParser(description='Train Chinese Word2Vec task')
parser.add_argument('--src_data', type=str, default='data/text', help='source data path')
parser.add_argument('--model_path', type=str, default='data/model', help='model save path')
parser.add_argument('--vector_path', type=str, default='data/vector', help='vector save path')

args = parser.parse_args()


'''
    Author： Snowy    Time：2018-10-29
    进行分词
    :param sentence 原始句子
    :return 切分好的句子
'''
def cut_words(sentence):
    return " ".join(jieba.cut(sentence))   #.encode('utf-8')

'''
    Author： Snowy    Time：2018-10-29
    进行分词
    :param src_file 原文件 格式是:一行一篇文章或者一个完整的片段
    :param target_file 目标文件 格式是:一行一篇文章或者一个完整的片段，已经切分好了词
    :return 
'''
def get_seg_file(src_file, target_file):
    # logger Part
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('runnung %s' % ' '.join(sys.argv))


    # 分词相关部分
    f = open(src_file, 'r', encoding="utf-8")
    target = open(target_file, 'w', encoding="utf-8")

    lines = f.readlines()

    line_num = 1
    for line in lines:
        # print ('************processing ', line_num , " article ************")
        line_seg = cut_words(line)
        target.writelines(str(line_seg))
        line_num += 1
        if (line_num % 10000 == 0):
            logger.info("Having Processed " + str(line_num) + " articles")

    f.close()
    target.close()
    logger.info("Finished CutWord " + str(line_num) + " articles")


# 训练word2vec   使用的语料是进行过分词的文本
def word2vec(seg_file, model_file, vector_file):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s " % ' '.join(sys.argv))

    model = Word2Vec(LineSentence(seg_file), size=50, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(model_file)
    model.wv.save_word2vec_format(vector_file, binary=False)





if __name__ == "__main__":

    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())
        sys.exit(1)

    # src_file, model_file, vector_file = sys.argv[1:4]

    src_file = args.src_data
    model_file = args.model_path
    vector_file = args.vector_path


    seg_file = src_file + ".seg"
    # 文本分词
    get_seg_file(src_file, seg_file)

    # 词向量的训练
    word2vec(seg_file, model_file, vector_file)


