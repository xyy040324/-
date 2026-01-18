#任务分配：
#孟祥政：nlp虚拟环境配置，接入豆包API配置，服务器运行AIQA配置，app.py编写，界面模块HTML编写，机器翻译模型训练
#唐俊杰：情感分析模型优化与训练（准确率0.9以上），界面模块HTML编写
#9班王天宇：文本分类模型优化与训练（准确率0.85以上），情感分析模型训练
#马康超：机器翻译模型优化与训练（准确率0.85以上），文本分类模型训练
# -*- coding: utf-8 -*-
"""
10.3.2 情感分析模型训练脚本
优化版本：改进代码结构、错误处理、添加回调函数和模型检查点
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import jieba
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# 设置字体（使用系统自带字体，避免 SimHei 缺失导致的警告）
import matplotlib
import logging
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 抑制 matplotlib 字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*findfont.*')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# ==================== 数据预处理函数 ====================
def load_sentiment_data(pos_path, neg_path, comment_path=None):
    """
    加载情感分析数据
    
    Args:
        pos_path: 正向情感数据路径
        neg_path: 负向情感数据路径
        comment_path: 评论数据路径（可选）
    
    Returns:
        pn_all: 合并后的数据框
    """
    print("正在加载情感数据...")
    
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"正向情感数据文件不存在: {pos_path}")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"负向情感数据文件不存在: {neg_path}")
    
    # 读取正负情感语料
    try:
        pos = pd.read_excel(pos_path, header=None, index_col=None)
        neg = pd.read_excel(neg_path, header=None, index_col=None)
    except Exception as e:
        raise ValueError(f"读取Excel文件失败: {e}")
    
    # 给训练语料贴标签
    pos['mark'] = 1
    neg['mark'] = 0
    
    # 合并正负情感语料
    pn_all = pd.concat([pos, neg], ignore_index=True)
    
    # 确保所有输入都是字符串类型
    pn_all[0] = pn_all[0].astype(str)
    
    # 过滤空数据
    pn_all = pn_all[pn_all[0].str.strip() != '']
    
    if len(pn_all) == 0:
        raise ValueError("合并后的数据为空")
    
    print(f"正向样本数: {len(pos)}, 负向样本数: {len(neg)}")
    
    # 如果提供了评论数据，也加入训练
    if comment_path and os.path.exists(comment_path):
        try:
            comment = pd.read_excel(comment_path)
            if 'rateContent' in comment.columns:
                comment = comment[comment['rateContent'].notnull()]
                print(f"额外评论数据: {len(comment)} 条")
                return pn_all, comment
        except Exception as e:
            print(f"警告: 读取评论数据失败: {e}")
    
    return pn_all, None


def build_vocabulary(pn_all, comment=None):
    """
    构建词汇表
    
    Args:
        pn_all: 正负情感数据框
        comment: 评论数据框（可选）
    
    Returns:
        dicts: 词汇表数据框
    """
    print("正在构建词汇表...")
    
    # 分词函数
    def cut_word(x):
        try:
            return list(jieba.cut(str(x)))
        except:
            return []
    
    # 对情感语料分词
    pn_all['words'] = pn_all[0].apply(cut_word)
    
    # 合并所有词语
    all_words = []
    for words in pn_all['words']:
        all_words.extend(words)
    
    # 如果提供了评论数据，也加入词汇表
    if comment is not None and 'rateContent' in comment.columns:
        comment['words'] = comment['rateContent'].apply(cut_word)
        for words in comment['words']:
            all_words.extend(words)
    
    if not all_words:
        raise ValueError("分词后没有有效词语")
    
    # 建立统计词典
    word_counts = pd.Series(all_words).value_counts()
    dicts = pd.DataFrame(word_counts)
    dicts['id'] = list(range(1, len(dicts) + 1))
    
    print(f"词汇表大小: {len(dicts)}")
    return dicts


def vectorize_texts(pn_all, dicts, maxlen=50):
    """
    将文本向量化
    
    Args:
        pn_all: 数据框
        dicts: 词汇表
        maxlen: 最大序列长度
    
    Returns:
        x_all: 向量化的文本
        y_all: 标签
    """
    print("正在向量化文本...")
    
    def get_sent(x):
        """将词语转换为ID"""
        ids = []
        for word in x:
            if word in dicts.index:
                ids.append(dicts.loc[word, 'id'])
        return ids
    
    # 将词语转换为ID序列
    pn_all['sent'] = pn_all['words'].apply(get_sent)
    
    # 过滤空序列
    pn_all = pn_all[pn_all['sent'].apply(len) > 0]
    
    if len(pn_all) == 0:
        raise ValueError("向量化后没有有效数据")
    
    # 对序列进行padding
    sequences = list(pn_all['sent'])
    x_all = sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    y_all = np.array(list(pn_all['mark']))
    
    print(f"向量化完成，数据形状: {x_all.shape}")
    return x_all, y_all


# ==================== 模型构建 ====================
def build_sentiment_model(vocab_size, maxlen=50, embedding_dim=256, lstm_units=128):
    """
    构建情感分析LSTM模型
    
    Args:
        vocab_size: 词汇表大小
        maxlen: 最大序列长度
        embedding_dim: 词向量维度
        lstm_units: LSTM单元数
    
    Returns:
        编译好的模型
    """
    model = Sequential([
        Input(shape=(maxlen,)),
        Embedding(vocab_size + 1, embedding_dim, mask_zero=True),
        LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ])
    
    return model


# ==================== 训练函数 ====================
def train_sentiment_model(
    pos_path, neg_path, comment_path=None,
    save_dir='../tmp/',
    maxlen=50,
    batch_size=16,
    epochs=10,
    embedding_dim=256,
    lstm_units=128,
    test_size=0.25,
    random_state=42
):
    """
    训练情感分析模型
    
    Args:
        pos_path: 正向情感数据路径
        neg_path: 负向情感数据路径
        comment_path: 评论数据路径（可选）
        save_dir: 模型保存目录
        maxlen: 最大序列长度
        batch_size: 批次大小
        epochs: 训练轮数
        embedding_dim: 词向量维度
        lstm_units: LSTM单元数
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        训练历史记录和测试结果
    """
    print("=" * 60)
    print("开始训练情感分析模型")
    print("=" * 60)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    pn_all, comment = load_sentiment_data(pos_path, neg_path, comment_path)
    
    # 构建词汇表
    dicts = build_vocabulary(pn_all, comment)
    
    # 保存词汇表（可选）
    vocab_path = os.path.join(save_dir, 'sentiment_vocab.pkl')
    try:
        dicts.to_pickle(vocab_path)
        print(f"词汇表已保存到: {vocab_path}")
    except:
        pass
    
    # 向量化文本
    x_all, y_all = vectorize_texts(pn_all, dicts, maxlen)
    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )
    
    print(f"训练集大小: {x_train.shape}")
    print(f"测试集大小: {x_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.2%}")
    print(f"测试集正样本比例: {y_test.mean():.2%}")
    
    # 构建模型
    vocab_size = len(dicts)
    print("正在构建模型...")
    model = build_sentiment_model(vocab_size, maxlen, embedding_dim, lstm_units)
    
    # 编译模型
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 设置回调函数
    model_path = os.path.join(save_dir, 'sentiment_best.h5')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 训练模型
    print("开始训练...")
    start_time = time.time()
    
    # 划分验证集
    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    history = model.fit(
        x_train_fit, y_train_fit,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"训练耗时: {int(training_time)} 秒")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'sentiment_final.h5')
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
    # 绘制训练过程
    plot_training_history(history, save_dir)
    
    # 测试模型
    print("\n正在测试模型...")
    test_results = evaluate_model(model, x_test, y_test, save_dir)
    
    return history, test_results, dicts


def plot_training_history(history, save_dir):
    """
    绘制训练过程图表
    
    Args:
        history: 训练历史记录
        save_dir: 保存目录
    """
    epochs = len(history.history['loss'])
    
    plt.figure(figsize=(12, 5))
    
    # 准确率趋势图
    plt.subplot(121)
    plt.title('准确率趋势图', fontsize=14)
    plt.plot(range(1, epochs + 1), history.history['accuracy'], 
             linestyle='-', color='g', label='训练集', marker='o')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], 
             linestyle='-.', color='b', label='验证集', marker='s')
    plt.legend(loc='best')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.grid(True, alpha=0.3)
    
    # 损失趋势图
    plt.subplot(122)
    plt.title('损失趋势图', fontsize=14)
    plt.plot(range(1, epochs + 1), history.history['loss'], 
             linestyle='-', color='g', label='训练集', marker='o')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], 
             linestyle='-.', color='b', label='验证集', marker='s')
    plt.legend(loc='best')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'sentiment_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图表已保存到: {plot_path}")
    plt.close()


def evaluate_model(model, x_test, y_test, save_dir):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        x_test: 测试集输入
        y_test: 测试集标签
        save_dir: 保存目录
    
    Returns:
        评估结果字典
    """
    # 预测
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    # 计算指标
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(metrics.classification_report(y_test, y_pred, target_names=['负向', '正向']))
    
    # 混淆矩阵
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负向', '正向'], yticklabels=['负向', '正向'])
    plt.xlabel('预测标签', size=14)
    plt.ylabel('真实标签', size=14)
    plt.title('混淆矩阵', size=16)
    plt.tight_layout()
    
    conf_matrix_path = os.path.join(save_dir, 'sentiment_confusion_matrix.png')
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {conf_matrix_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置路径（根据当前脚本位置自动定位 data/tmp 目录）
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(cur_dir, '../data/')
    pos_path = os.path.join(base_dir, 'pos.xls')
    neg_path = os.path.join(base_dir, 'neg.xls')
    comment_path = os.path.join(base_dir, 'sum.xls')
    save_dir = os.path.join(cur_dir, '../tmp/')
    
    try:
        # 训练模型
        history, test_results, dicts = train_sentiment_model(
            pos_path=pos_path,
            neg_path=neg_path,
            comment_path=comment_path,
            save_dir=save_dir,
            maxlen=50,
            batch_size=16,
            epochs=10
        )
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"测试集准确率: {test_results['accuracy']:.4f}")
        print(f"F1值: {test_results['f1']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
