#任务分配：
#孟祥政：nlp虚拟环境配置，接入豆包API配置，服务器运行AIQA配置，app.py编写，界面模块HTML编写，机器翻译模型训练
#唐俊杰：情感分析模型优化与训练（准确率0.9以上），界面模块HTML编写
#9班王天宇：文本分类模型优化与训练（准确率0.85以上），情感分析模型训练
#马康超：机器翻译模型优化与训练（准确率0.85以上），文本分类模型训练
# -*- coding: utf-8 -*-
import os
import sys
import http.client
import json
import time
import threading
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import jieba
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
# 修改这里：使用tf.keras而不是单独的keras
from tensorflow.keras.models import load_model

# 导入训练模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlp_deeplearn', 'code'))
try:
    from importlib import import_module
    # 动态导入训练模块
    _training_modules = {}
except:
    pass

# --------------------------- 路径与全局资源 --------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR = os.path.join(BASE_DIR, 'nlp_deeplearn')
DATA_DIR = os.path.join(NLP_DIR, 'data')
TMP_DIR = os.path.join(NLP_DIR, 'tmp')

# 文本分类
_cls_model: Optional[tf.keras.Model] = None
_cls_word_to_id: Optional[Dict[str, int]] = None
_cls_categories: List[str] = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
_CLS_SEQ_LEN = 600

# 情感分析
_sent_model: Optional[tf.keras.Model] = None
_sent_dicts: Optional[pd.DataFrame] = None
_SENT_MAXLEN = 50

# 机器翻译
_enc_model = None
_dec_model = None
_inp_lang_tokenizer = None
_targ_lang_tokenizer = None
_max_length_targ = None
_max_length_inp = None

# 训练状态
_training_status = {
    'text_classification': {'status': 'idle', 'progress': 0, 'message': ''},
    'sentiment': {'status': 'idle', 'progress': 0, 'message': ''},
    'translation': {'status': 'idle', 'progress': 0, 'message': ''}
}


def _safe_path(*paths: str) -> str:
    """构建绝对路径，避免相对路径问题。"""
    return os.path.join(BASE_DIR, *paths)


# ---------- 文本分类：加载与预测 ---------- #
def _load_text_classification_assets():
    global _cls_model, _cls_word_to_id
    if _cls_model is not None and _cls_word_to_id is not None:
        return

    vocab_path = _safe_path('nlp_deeplearn', 'data', 'cnews.vocab.txt')
    # 尝试多个可能的模型路径
    model_paths = [
        _safe_path('nlp_deeplearn', 'tmp', 'text_classification_best.h5'),
        _safe_path('nlp_deeplearn', 'tmp', 'text_classification_final.h5'),
        _safe_path('nlp_deeplearn', 'tmp', 'my_model.h5')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not os.path.exists(vocab_path) or model_path is None:
        raise FileNotFoundError('文本分类模型或词表未找到，请先训练并保存。')

    with open(vocab_path, encoding='utf-8') as f:
        words = [line.strip() for line in f.readlines()]
    _cls_word_to_id = dict(zip(words, range(len(words))))
    _cls_model = load_model(model_path)


def classify_text(text: str) -> str:
    """对中文文本进行10类新闻分类。"""
    _load_text_classification_assets()
    ids = [_cls_word_to_id[ch] for ch in text if ch in _cls_word_to_id]
    if not ids:
        return '内容过短或无有效词汇'
    x_pad = sequence.pad_sequences([ids], _CLS_SEQ_LEN)
    prob = _cls_model.predict(x_pad, verbose=0)
    idx = int(np.argmax(prob, axis=1)[0])
    return _cls_categories[idx]


# ---------- 情感分析：加载与预测 ---------- #
def _build_sentiment_dicts() -> pd.DataFrame:
    """复现10_3_2.py中的词典构建流程，保证与训练一致。"""
    # 使用_safe_path确保路径正确
    pos_path = _safe_path('nlp_deeplearn', 'data', 'pos.xls')
    neg_path = _safe_path('nlp_deeplearn', 'data', 'neg.xls')
    sum_path = _safe_path('nlp_deeplearn', 'data', 'sum.xls')
    
    pos = pd.read_excel(pos_path, header=None, index_col=None)
    neg = pd.read_excel(neg_path, header=None, index_col=None)
    pos['mark'] = 1
    neg['mark'] = 0
    pn_all = pd.concat([pos, neg], ignore_index=True)
    pn_all[0] = pn_all[0].astype(str)
    cut_word = lambda x: list(jieba.cut(x))
    pn_all['words'] = pn_all[0].apply(cut_word)
    comment = pd.read_excel(sum_path)
    comment = comment[comment['rateContent'].notnull()]
    comment['words'] = comment['rateContent'].apply(cut_word)
    pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)
    w: List[str] = []
    for i in pn_comment:
        w.extend(i)
    dicts = pd.DataFrame(pd.Series(w).value_counts())
    dicts['id'] = list(range(1, len(dicts) + 1))
    return dicts


def _load_sentiment_assets():
    global _sent_model, _sent_dicts
    if _sent_model is not None and _sent_dicts is not None:
        return
    # 尝试多个可能的模型路径
    model_paths = [
        _safe_path('nlp_deeplearn', 'tmp', 'sentiment_best.h5'),
        _safe_path('nlp_deeplearn', 'tmp', 'sentiment_final.h5'),
        _safe_path('nlp_deeplearn', 'tmp', 'sentiment_model.h5')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError('情感分析模型未找到，请先训练并保存。')
    _sent_dicts = _build_sentiment_dicts()
    _sent_model = load_model(model_path)


def _sent_to_ids(text: str) -> List[int]:
    cut_word = lambda x: list(jieba.cut(x))
    words = cut_word(str(text))
    ids = []
    for word in words:
        if word in _sent_dicts['id']:
            ids.append(_sent_dicts['id'][word])
    # 处理空列表的情况
    if not ids:
        ids = [0]
    padded = sequence.pad_sequences([ids], maxlen=_SENT_MAXLEN)[0]
    return list(padded)


def sentiment_predict(text: str) -> Tuple[str, float]:
    """返回情感标签与置信度。"""
    _load_sentiment_assets()
    ids = _sent_to_ids(text)
    pred = _sent_model.predict(np.array([ids]), verbose=0)[0][0]
    label = '正向' if pred >= 0.5 else '负向'
    return label, float(pred)


# ---------- 机器翻译：加载与预测 ---------- #
def _preprocess_sentence(w: str) -> str:
    import re
    w = re.sub(r'([?.!,])', r' \1 ', w)
    w = re.sub(r"[' ']+", ' ', w)
    return '<start> ' + w + ' <end>'


def _tokenize(lang: List[str]):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def _create_dataset(path: str, num_examples=None):
    import io
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[_preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return list(zip(*word_pairs))


def _load_translation_assets():
    """加载词表与模型检查点，仅用于推理。"""
    global _enc_model, _dec_model, _inp_lang_tokenizer, _targ_lang_tokenizer
    global _max_length_targ, _max_length_inp
    if all([_enc_model, _dec_model, _inp_lang_tokenizer, _targ_lang_tokenizer, _max_length_targ, _max_length_inp]):
        return

    path_to_file = _safe_path('nlp_deeplearn', 'data', 'en-ch.txt')
    checkpoint_dir = _safe_path('nlp_deeplearn', 'tmp', 'training_checkpoints')
    if not os.path.exists(path_to_file) or not os.path.exists(checkpoint_dir):
        raise FileNotFoundError('机器翻译数据或检查点未找到，请先训练并保存。')

    targ_lang, inp_lang = _create_dataset(path_to_file, 2000)
    input_tensor, _inp_lang_tokenizer = _tokenize(inp_lang)
    target_tensor, _targ_lang_tokenizer = _tokenize(targ_lang)
    _max_length_targ = max(len(t) for t in target_tensor)
    _max_length_inp = max(len(t) for t in input_tensor)

    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(_inp_lang_tokenizer.word_index) + 1
    vocab_tar_size = len(_targ_lang_tokenizer.word_index) + 1

    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units):
            super().__init__()
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state=hidden)
            return output, state

        def initialize_hidden_state(self, batch_size):
            return tf.zeros((batch_size, self.enc_units))

    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super().__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            hidden_with_time_axis = tf.expand_dims(query, 1)
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights

    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units):
            super().__init__()
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)
            self.attention = BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            context_vector, attention_weights = self.attention(hidden, enc_output)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(x)
            output = tf.reshape(output, (-1, output.shape[2]))
            x = self.fc(output)
            return x, state, attention_weights

    _enc_model = Encoder(vocab_inp_size, embedding_dim, units)
    _dec_model = Decoder(vocab_tar_size, embedding_dim, units)

    ckpt = tf.train.Checkpoint(encoder=_enc_model, decoder=_dec_model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest:
        raise FileNotFoundError('未找到翻译模型检查点。')
    ckpt.restore(latest).expect_partial()


def translate_sentence(sentence: str) -> str:
    """使用已训练的Seq2Seq模型进行中->英翻译。"""
    _load_translation_assets()
    sentence = _preprocess_sentence(sentence)
    inputs = [_inp_lang_tokenizer.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    hidden = _enc_model.initialize_hidden_state(batch_size=1)
    enc_out, enc_hidden = _enc_model(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([_targ_lang_tokenizer.word_index['<start>']], 0)

    result_words = []
    for _ in range(_max_length_targ):
        predictions, dec_hidden, _ = _dec_model(dec_input, dec_hidden, enc_out)
        predicted_id = int(tf.argmax(predictions[0]).numpy())
        predicted_word = _targ_lang_tokenizer.index_word.get(predicted_id, '')
        if predicted_word == '<end>':
            break
        if predicted_word and predicted_word != '<start>':
            result_words.append(predicted_word)
        dec_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result_words) if result_words else '无法翻译'


# 确保templates文件夹存在
if not os.path.exists('templates'):
    os.makedirs('templates')
    print("已创建 templates 文件夹")

# 豆包API调用函数
def call_doubao_api(user_message, system_prompt="你是一个有用的助手。"):
    """
    调用豆包API进行对话
    
    Args:
        user_message: 用户输入的消息
        system_prompt: 系统提示词，默认是助手角色
        
    Returns:
        API返回的响应文本
    """
    conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")
    
    payload = json.dumps({
        "model": "doubao-seed-1-6-vision-250815",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False
    })
    
    headers = {
        'Authorization': 'Bearer dbf0a80b-e5b4-470d-b667-f63084e21443',
        'Content-Type': 'application/json',
        'User-Agent': 'Flask-Doubao-Chat/1.0'
    }
    
    try:
        print(f"发送请求到豆包API: {user_message[:50]}...")
        start_time = time.time()
        
        conn.request("POST", "/api/v3/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        end_time = time.time()
        print(f"API响应时间: {end_time - start_time:.2f}秒")
        print(f"API响应状态: {res.status}")
        
        response_json = json.loads(data.decode("utf-8"))
        
        # 调试：打印完整的响应结构
        if res.status != 200:
            print(f"API错误响应: {response_json}")
            return f"API请求失败，状态码: {res.status}"
        
        # 提取回复内容
        if "choices" in response_json and len(response_json["choices"]) > 0:
            reply = response_json["choices"][0]["message"]["content"]
            print(f"获取到回复: {reply[:50]}...")
            return reply
        elif "error" in response_json:
            error_msg = response_json["error"].get("message", "未知错误")
            print(f"API返回错误: {error_msg}")
            return f"抱歉，API返回错误: {error_msg}"
        else:
            print(f"意外的响应格式: {response_json}")
            return "抱歉，我暂时无法回答这个问题。"
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return "抱歉，服务响应格式错误。"
    except Exception as e:
        print(f"API调用出错: {e}")
        return "抱歉，服务暂时不可用。"
    finally:
        conn.close()

# 实例化Flask应用
app = Flask(__name__, static_url_path='/static')


def preload_nlp_models():
    """预加载所有NLP模型"""
    try:
        print("开始预加载NLP模型...")
        
        # 文本分类模型
        print(" 加载文本分类模型...")
        _load_text_classification_assets()
        print("  文本分类模型加载完成")
        
        # 情感分析模型
        print(" 加载情感分析模型...")
        _load_sentiment_assets()
        print("  情感分析模型加载完成")
        
        # 机器翻译模型
        print(" 加载机器翻译模型...")
        _load_translation_assets()
        print("  机器翻译模型加载完成")
        
        print("所有NLP模型预加载完成!")
    except Exception as e:
        print(f"模型预加载失败: {e}")
        print("注意: NLP功能可能无法正常工作")


@app.route('/message', methods=['POST'])
def reply():
    """
    处理用户消息并返回豆包API的回复
    """
    try:
        # 获取用户输入
        req_msg = request.form.get('msg', '').strip()
        
        # 检查输入是否为空
        if not req_msg:
            return jsonify({'text': '请输入一些内容吧~'})
        
        print(f"收到用户消息: {req_msg}")
        
        # 调用豆包API获取回复
        res_msg = call_doubao_api(req_msg)
        
        # 清理和格式化回复
        res_msg = res_msg.strip()
        if not res_msg:
            res_msg = '我们来聊聊天吧'
        
        return jsonify({'text': res_msg})
        
    except Exception as e:
        print(f"处理消息时出错: {e}")
        return jsonify({'text': '抱歉，处理您的消息时出现错误。'})


@app.route("/")
def index():
    """
    返回聊天界面
    """
    print("访问首页")
    return render_template('index.html')


@app.route('/test', methods=['GET'])
def test_api():
    """
    测试API是否正常工作
    """
    try:
        test_message = "你好，请介绍一下你自己。"
        print(f"测试API，发送消息: {test_message}")
        
        response = call_doubao_api(test_message)
        
        result = {
            'status': 'success',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_message': test_message,
            'response': response,
            'service': 'Doubao Chat API'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"测试API时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    try:
        # 简单测试API连接
        test_response = call_doubao_api("健康检查", "你只需要回复'健康检查通过'即可")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'service': 'Doubao Chat API',
            'version': '1.0',
            'api_status': 'connected',
            'test_response': test_response[:100] if test_response else '无响应'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })


# --------------------------- NLP 推理接口 --------------------------- #
@app.route('/nlp/classify', methods=['POST'])
def api_classify():
    """新闻文本分类接口。"""
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'text 不能为空'}), 400
    try:
        category = classify_text(text)
        return jsonify({'status': 'success', 'category': category})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/nlp/sentiment', methods=['POST'])
def api_sentiment():
    """情感分析接口。"""
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'text 不能为空'}), 400
    try:
        label, score = sentiment_predict(text)
        return jsonify({'status': 'success', 'label': label, 'score': score})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/nlp/translate', methods=['POST'])
def api_translate():
    """Seq2Seq 机器翻译接口（中->英）。"""
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'text 不能为空'}), 400
    try:
        result = translate_sentence(text)
        return jsonify({'status': 'success', 'translation': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --------------------------- NLP 训练接口 --------------------------- #
def _train_text_classification_async():
    """异步训练文本分类模型"""
    try:
        _training_status['text_classification']['status'] = 'training'
        _training_status['text_classification']['message'] = '开始训练文本分类模型...'
        
        # 导入训练模块
        code_dir = _safe_path('nlp_deeplearn', 'code')
        sys.path.insert(0, code_dir)
        from importlib import import_module
        train_module = import_module('10_3_1')
        
        # 设置路径
        base_dir = _safe_path('nlp_deeplearn', 'data')
        train_dir = os.path.join(base_dir, 'cnews.train.txt')
        val_dir = os.path.join(base_dir, 'cnews.val.txt')
        test_dir = os.path.join(base_dir, 'cnews.test.txt')
        vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
        save_dir = _safe_path('nlp_deeplearn', 'tmp')
        
        _training_status['text_classification']['message'] = '正在加载数据...'
        
        # 调用训练函数
        history, test_results = train_module.train_text_classification_model(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            vocab_dir=vocab_dir,
            save_dir=save_dir,
            vocab_size=5000,
            seq_length=600,
            batch_size=64,
            epochs=20
        )
        
        _training_status['text_classification']['status'] = 'completed'
        _training_status['text_classification']['progress'] = 100
        _training_status['text_classification']['message'] = f'训练完成！测试集准确率: {test_results["accuracy"]:.4f}'
        
        # 重新加载模型
        global _cls_model, _cls_word_to_id
        _cls_model = None
        _cls_word_to_id = None
        
    except Exception as e:
        _training_status['text_classification']['status'] = 'error'
        _training_status['text_classification']['message'] = f'训练失败: {str(e)}'
        import traceback
        traceback.print_exc()


def _train_sentiment_async():
    """异步训练情感分析模型"""
    try:
        _training_status['sentiment']['status'] = 'training'
        _training_status['sentiment']['message'] = '开始训练情感分析模型...'
        
        # 导入训练模块
        code_dir = _safe_path('nlp_deeplearn', 'code')
        sys.path.insert(0, code_dir)
        from importlib import import_module
        train_module = import_module('10_3_2')
        
        # 设置路径
        base_dir = _safe_path('nlp_deeplearn', 'data')
        pos_path = os.path.join(base_dir, 'pos.xls')
        neg_path = os.path.join(base_dir, 'neg.xls')
        comment_path = os.path.join(base_dir, 'sum.xls')
        save_dir = _safe_path('nlp_deeplearn', 'tmp')
        
        _training_status['sentiment']['message'] = '正在加载数据...'
        
        # 调用训练函数
        history, test_results, dicts = train_module.train_sentiment_model(
            pos_path=pos_path,
            neg_path=neg_path,
            comment_path=comment_path,
            save_dir=save_dir,
            maxlen=50,
            batch_size=16,
            epochs=10
        )
        
        _training_status['sentiment']['status'] = 'completed'
        _training_status['sentiment']['progress'] = 100
        _training_status['sentiment']['message'] = f'训练完成！测试集准确率: {test_results["accuracy"]:.4f}'
        
        # 重新加载模型
        global _sent_model, _sent_dicts
        _sent_model = None
        _sent_dicts = None
        
    except Exception as e:
        _training_status['sentiment']['status'] = 'error'
        _training_status['sentiment']['message'] = f'训练失败: {str(e)}'
        import traceback
        traceback.print_exc()


def _train_translation_async():
    """异步训练机器翻译模型"""
    try:
        _training_status['translation']['status'] = 'training'
        _training_status['translation']['message'] = '开始训练机器翻译模型...'
        
        # 导入训练模块
        code_dir = _safe_path('nlp_deeplearn', 'code')
        sys.path.insert(0, code_dir)
        from importlib import import_module
        train_module = import_module('10_4')
        
        # 设置路径
        data_path = _safe_path('nlp_deeplearn', 'data', 'en-ch.txt')
        save_dir = _safe_path('nlp_deeplearn', 'tmp')
        
        _training_status['translation']['message'] = '正在加载数据...'
        
        # 调用训练函数
        results = train_module.train_translation_model(
            data_path=data_path,
            save_dir=save_dir,
            num_examples=2000,
            batch_size=64,
            embedding_dim=256,
            units=1024,
            epochs=50
        )
        
        _training_status['translation']['status'] = 'completed'
        _training_status['translation']['progress'] = 100
        _training_status['translation']['message'] = f'训练完成！最佳验证损失: {results["best_val_loss"]:.4f}'
        
        # 重新加载模型
        global _enc_model, _dec_model, _inp_lang_tokenizer, _targ_lang_tokenizer
        global _max_length_targ, _max_length_inp
        _enc_model = None
        _dec_model = None
        _inp_lang_tokenizer = None
        _targ_lang_tokenizer = None
        _max_length_targ = None
        _max_length_inp = None
        
    except Exception as e:
        _training_status['translation']['status'] = 'error'
        _training_status['translation']['message'] = f'训练失败: {str(e)}'
        import traceback
        traceback.print_exc()


@app.route('/nlp/train/text_classification', methods=['POST'])
def api_train_text_classification():
    """训练文本分类模型接口"""
    if _training_status['text_classification']['status'] == 'training':
        return jsonify({
            'status': 'error',
            'message': '模型正在训练中，请稍候...'
        }), 400
    
    try:
        # 重置状态
        _training_status['text_classification'] = {'status': 'training', 'progress': 0, 'message': '开始训练...'}
        
        # 在后台线程中训练
        thread = threading.Thread(target=_train_text_classification_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '训练任务已启动，请在 /nlp/train/status 查看训练状态'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/nlp/train/sentiment', methods=['POST'])
def api_train_sentiment():
    """训练情感分析模型接口"""
    if _training_status['sentiment']['status'] == 'training':
        return jsonify({
            'status': 'error',
            'message': '模型正在训练中，请稍候...'
        }), 400
    
    try:
        # 重置状态
        _training_status['sentiment'] = {'status': 'training', 'progress': 0, 'message': '开始训练...'}
        
        # 在后台线程中训练
        thread = threading.Thread(target=_train_sentiment_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '训练任务已启动，请在 /nlp/train/status 查看训练状态'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/nlp/train/translation', methods=['POST'])
def api_train_translation():
    """训练机器翻译模型接口"""
    if _training_status['translation']['status'] == 'training':
        return jsonify({
            'status': 'error',
            'message': '模型正在训练中，请稍候...'
        }), 400
    
    try:
        # 重置状态
        _training_status['translation'] = {'status': 'training', 'progress': 0, 'message': '开始训练...'}
        
        # 在后台线程中训练
        thread = threading.Thread(target=_train_translation_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '训练任务已启动，请在 /nlp/train/status 查看训练状态'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/nlp/train/status', methods=['GET'])
def api_training_status():
    """获取训练状态接口"""
    model_type = request.args.get('model', 'all')
    
    if model_type == 'all':
        return jsonify({
            'status': 'success',
            'training_status': _training_status
        })
    elif model_type in _training_status:
        return jsonify({
            'status': 'success',
            'model': model_type,
            'training_status': _training_status[model_type]
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'未知的模型类型: {model_type}'
        }), 400


@app.errorhandler(404)
def not_found(error):
    """
    处理404错误
    """
    return jsonify({
        'status': 'error',
        'message': '请求的页面不存在',
        'error': str(error)
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """
    处理500错误
    """
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误',
        'error': str(error)
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 启动豆包聊天机器人服务")
    print("=" * 60)
    
    # 预加载NLP模型
    preload_nlp_models()
    
    # 检查模板文件
    index_path = os.path.join('templates', 'index.html')
    if os.path.exists(index_path):
        print(f"✅ 找到模板文件: {index_path}")
    else:
        print(f"❌ 未找到模板文件: {index_path}")
        print("请确保 templates/index.html 文件存在")
    
    print("\n📡 访问地址:")
    print("   聊天界面: http://127.0.0.1:8808")
    print("   API测试: http://127.0.0.1:8808/test")
    print("   健康检查: http://127.0.0.1:8808/health")
    print("\n📝 API接口:")
    print("   POST /message - 发送和接收消息")
    print("   POST /nlp/classify - 文本分类")
    print("   POST /nlp/sentiment - 情感分析")
    print("   POST /nlp/translate - 机器翻译")
    print("\n🔧 训练接口:")
    print("   POST /nlp/train/text_classification - 训练文本分类模型")
    print("   POST /nlp/train/sentiment - 训练情感分析模型")
    print("   POST /nlp/train/translation - 训练机器翻译模型")
    print("   GET  /nlp/train/status?model=<model_type> - 查看训练状态")
    print("=" * 60)
    
    # 启动Flask应用
    app.run(
        host='127.0.0.1',
        port=8808,
        debug=True,
        threaded=True  # 启用多线程
    )