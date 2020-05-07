from flask import Flask,make_response,render_template,request
import tensorflow as tf
from models import *

app = Flask(__name__)



print("model loading")
# 导入模型
myModel = my_densenet()
# 加载训练好的参数
myModel.load_weights("./static/model/model.h5")
print("model loaded")

CODE_CLASS_MAP = {
    0: '猫',
    1: '狗',
}


@app.route('/',methods=['GET'])
def index():
    res = make_response(render_template('index1.html'))
    return res

@app.route('/uploadImg/', methods=['POST'])
def pets_classify():
    """
    宠物图片分类接口，上传一张图片，返回此图片上的宠物是那种类别，概率多少
    """
    # 获取用户上传的图片
    img_str = request.files.get('file').read()
    # 进行数据预处理
    x = tf.image.decode_image(img_str, channels=3)
    x = tf.image.resize(x, (224, 224))
    x = x / 255.
    x = (x - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.299, 0.224, 0.225])
    x = tf.reshape(x, (1, 224, 224, 3))
    # 预测
    y_pred = myModel(x)
    pet_cls_code = tf.argmax(y_pred, axis=1).numpy()[0]
    pet_class = CODE_CLASS_MAP.get(pet_cls_code)

    res = make_response(render_template("index2.html",pet = pet_class))
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)

