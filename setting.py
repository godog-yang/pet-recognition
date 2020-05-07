
# 图片类别和搜索关键词的映射关系
IMAGE_CLASS_KEYWORD_MAP = {
    'cats': '猫',
    'dogs': '狗',
}
# 图片保存根目录
IMAGES_ROOT = './images'

# 每个类别选取的图片数量
SAMPLES_PER_CLASS = 400
# 参与训练的类别
CLASSES = ['cats', 'dogs']
# 参与训练的类别数量
CLASS_NUM = len(CLASSES)
# 类别->编号的映射
CLASS_CODE_MAP = {
    'cats': 0,
    'dogs': 1,
}
# 编号->类别的映射
CODE_CLASS_MAP = {
    0: '猫',
    1: '狗',
}
# 随机数种子
RANDOM_SEED = 25  # 两个类别时样本较为均衡的随机数种子

# 训练集比例
TRAIN_DATASET = 0.6
# 开发集比例
DEV_DATASET = 0.2
# 测试集比例
TEST_DATASET = 0.2
# mini_batch大小
BATCH_SIZE = 16
# imagenet数据集均值
IMAGE_MEAN = [0.485, 0.456, 0.406]
# imagenet数据集标准差
IMAGE_STD = [0.299, 0.224, 0.225]

# 学习率
LEARNING_RATE = 0.001
# 训练epoch数
TRAIN_EPOCHS = 30
# 保存训练模型的路径
MODEL_PATH = './static/model/model.h5'
