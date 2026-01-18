#任务分配：
#孟祥政：nlp虚拟环境配置，接入豆包API配置，服务器运行AIQA配置，app.py编写，界面模块HTML编写，机器翻译模型训练
#唐俊杰：情感分析模型优化与训练（准确率0.9以上），界面模块HTML编写
#9班王天宇：文本分类模型优化与训练（准确率0.85以上），情感分析模型训练
#马康超：机器翻译模型优化与训练（准确率0.85以上），文本分类模型训练
# -*- coding: utf-8 -*-
"""
10.3.1 文本分类模型训练脚本
优化版本：改进代码结构、错误处理、添加回调函数和模型检查点
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据预处理函数 ====================
def open_file(filename, mode='r'):
    """
    安全打开文件
    
    Args:
        filename: 文件路径
        mode: 打开模式 'r' 或 'w'
    
    Returns:
        文件对象
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """
    读取训练/测试文件数据
    
    Args:
        filename: 文件路径
    
    Returns:
        contents: 文本内容列表
        labels: 标签列表
    """
    contents, labels = [], []
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    with open_file(filename) as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                label, content = parts[0], parts[1]
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    if not contents:
        raise ValueError(f"文件 {filename} 中没有有效数据")
    
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    构建词汇表
    
    Args:
        train_dir: 训练集文件路径
        vocab_dir: 词汇表保存路径
        vocab_size: 词汇表大小
    """
    print(f"正在构建词汇表，大小: {vocab_size}")
    data_train, _ = read_file(train_dir)
    
    all_data = []
    for content in data_train:
        all_data.extend(content)
    
    if not all_data:
        raise ValueError("训练数据为空，无法构建词汇表")
    
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(vocab_dir), exist_ok=True)
    
    with open_file(vocab_dir, mode='w') as f:
        f.write('\n'.join(words) + '\n')
    
    print(f"词汇表已保存到: {vocab_dir}")


def read_vocab(vocab_dir):
    """
    读取词汇表
    
    Args:
        vocab_dir: 词汇表文件路径
    
    Returns:
        words: 词汇列表
        word_to_id: 词汇到ID的映射字典
    """
    if not os.path.exists(vocab_dir):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_dir}")
    
    with open_file(vocab_dir) as fp:
        words = [line.strip() for line in fp.readlines() if line.strip()]
    
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """
    读取分类目录
    
    Returns:
        categories: 类别列表
        cat_to_id: 类别到ID的映射字典
    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """
    将文件转换为ID表示
    
    Args:
        filename: 文件路径
        word_to_id: 词汇到ID的映射
        cat_to_id: 类别到ID的映射
        max_length: 最大序列长度
    
    Returns:
        x_pad: 填充后的输入序列
        y_pad: 独热编码的标签
    """
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    
    for i in range(len(contents)):
        # 将字符转换为ID，过滤不在词汇表中的字符
        ids = [word_to_id[x] for x in contents[i] if x in word_to_id]
        if not ids:  # 如果所有字符都不在词汇表中，跳过
            continue
        
        # 检查标签是否在类别字典中
        if labels[i] not in cat_to_id:
            print(f"警告: 未知标签 '{labels[i]}'，跳过")
            continue
        
        data_id.append(ids)
        label_id.append(cat_to_id[labels[i]])
    
    if not data_id:
        raise ValueError(f"文件 {filename} 处理后没有有效数据")
    
    # 使用Keras提供的pad_sequences将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, maxlen=max_length, padding='post', truncating='post')
    # 将标签转为独热编码（one-hot）表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    
    return x_pad, y_pad


# ==================== 模型构建 ====================
def build_text_classification_model(vocab_size, seq_length=600, embedding_dim=128, lstm_units=128):
    """
    构建文本分类LSTM模型
    
    Args:
        vocab_size: 词汇表大小
        seq_length: 序列长度
        embedding_dim: 词向量维度
        lstm_units: LSTM单元数
    
    Returns:
        编译好的模型
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=seq_length),
        tf.keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model


# ==================== 训练函数 ====================
def train_text_classification_model(
    train_dir, val_dir, test_dir, vocab_dir, 
    save_dir='../tmp/',
    vocab_size=5000,
    seq_length=600,
    batch_size=64,
    epochs=20,
    embedding_dim=128,
    lstm_units=128,
    use_multi_gpu=False
):
    """
    训练文本分类模型
    
    Args:
        train_dir: 训练集路径
        val_dir: 验证集路径
        test_dir: 测试集路径
        vocab_dir: 词汇表路径
        save_dir: 模型保存目录
        vocab_size: 词汇表大小
        seq_length: 序列长度
        batch_size: 批次大小
        epochs: 训练轮数
        embedding_dim: 词向量维度
        lstm_units: LSTM单元数
        use_multi_gpu: 是否使用多GPU
    
    Returns:
        训练历史记录和测试结果
    """
    print("=" * 60)
    print("开始训练文本分类模型")
    print("=" * 60)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建词汇表（如果不存在）
    if not os.path.exists(vocab_dir):
        print("词汇表不存在，正在构建...")
        build_vocab(train_dir, vocab_dir, vocab_size)
    
    # 读取分类目录和词汇表
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    vocab_size = len(words)
    print(f"词汇表大小: {vocab_size}")
    
    # 加载数据
    print("正在加载训练数据...")
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
    print(f"训练集大小: {x_train.shape}")
    print(f"训练集类别分布: {np.sum(y_train, axis=0)}")  # 显示每个类别的样本数
    
    print("正在加载验证数据...")
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
    print(f"验证集大小: {x_val.shape}")
    print(f"验证集类别分布: {np.sum(y_val, axis=0)}")  # 显示每个类别的样本数
    
    print("正在加载测试数据...")
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)
    print(f"测试集大小: {x_test.shape}")
    print(f"测试集类别分布: {np.sum(y_test, axis=0)}")  # 显示每个类别的样本数
    
    # 检查类别覆盖情况
    train_classes = np.unique(np.argmax(y_train, axis=1))
    val_classes = np.unique(np.argmax(y_val, axis=1))
    test_classes = np.unique(np.argmax(y_test, axis=1))
    all_classes = set(range(len(categories)))
    
    print(f"\n训练集包含类别: {train_classes}")
    print(f"验证集包含类别: {val_classes}")
    print(f"测试集包含类别: {test_classes}")
    print(f"所有10个类别: {sorted(all_classes)}")
    print(f"缺失的类别: {sorted(all_classes - set(train_classes) - set(val_classes) - set(test_classes))}")
    
    # 构建模型
    print("正在构建模型...")
    if use_multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = build_text_classification_model(vocab_size, seq_length, embedding_dim, lstm_units)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['categorical_accuracy']
            )
    else:
        model = build_text_classification_model(vocab_size, seq_length, embedding_dim, lstm_units)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['categorical_accuracy']
        )
    
    model.summary()
    
    # 设置回调函数
    model_path = os.path.join(save_dir, 'text_classification_best.h5')
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
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'text_classification_final.h5')
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
    # 绘制训练过程
    plot_training_history(history, save_dir)
    
    # 生成综合混淆矩阵（合并所有数据集，只输出一个图片）
    print("\n" + "="*60)
    print("生成综合混淆矩阵（合并训练集+验证集+测试集，包含所有10个类别）")
    print("="*60)
    
    # 获取脚本目录，用于保存1.png
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '1.png')
    
    generate_combined_confusion_matrix(model, x_train, y_train, x_val, y_val, 
                                       x_test, y_test, categories, output_path)
    
    # 计算各数据集的准确率（不生成图片，只打印）
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    
    # 测试集准确率
    y_test_pred = model.predict(x_test, verbose=0)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = np.sum(y_test_pred_classes == y_test_true_classes) / len(y_test_true_classes)
    
    # 验证集准确率
    y_val_pred = model.predict(x_val, verbose=0)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true_classes = np.argmax(y_val, axis=1)
    val_accuracy = np.sum(y_val_pred_classes == y_val_true_classes) / len(y_val_true_classes)
    
    # 训练集准确率
    y_train_pred = model.predict(x_train, verbose=0)
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)
    y_train_true_classes = np.argmax(y_train, axis=1)
    train_accuracy = np.sum(y_train_pred_classes == y_train_true_classes) / len(y_train_true_classes)
    
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    test_results = {
        'accuracy': test_accuracy,
        'confusion_matrix': None,
        'classification_report': None
    }
    
    return history, test_results


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
    plt.plot(range(1, epochs + 1), history.history['categorical_accuracy'], 
             linestyle='-', color='g', label='训练集', marker='o')
    plt.plot(range(1, epochs + 1), history.history['val_categorical_accuracy'], 
             linestyle='-.', color='b', label='验证集', marker='s')
    plt.legend(loc='best')
    x_major_locator = MultipleLocator(max(1, epochs // 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
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
    x_major_locator = MultipleLocator(max(1, epochs // 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'text_classification_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图表已保存到: {plot_path}")
    plt.close()


def generate_combined_confusion_matrix(model, x_train, y_train, x_val, y_val, 
                                       x_test, y_test, categories, output_path):
    """
    生成综合混淆矩阵（合并所有数据集），保存为1.png
    
    Args:
        model: 训练好的模型
        x_train, y_train: 训练集
        x_val, y_val: 验证集
        x_test, y_test: 测试集
        categories: 类别列表
        output_path: 输出文件路径（应为1.png的完整路径）
    """
    # 合并所有数据集
    all_x = np.vstack([x_train, x_val, x_test])
    all_y = np.vstack([y_train, y_val, y_test])
    
    print(f"合并数据集总大小: {len(all_x)}")
    print(f"  - 训练集样本数: {len(x_train)}")
    print(f"  - 验证集样本数: {len(x_val)}")
    print(f"  - 测试集样本数: {len(x_test)}")
    
    # 预测
    print("正在预测所有数据...")
    y_pred = model.predict(all_x, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(all_y, axis=1)
    
    # 计算综合混淆矩阵（确保包含所有10个类别）
    # confusion_matrix(y_true, y_pred): 行=真实标签，列=预测标签
    confm = confusion_matrix(y_true_classes, y_pred_classes, labels=list(range(len(categories))))
    
    # 按照参考图格式：X轴=真实标签，Y轴=预测标签
    # 需要转置矩阵以使格式匹配
    confm_transposed = confm.T
    
    # 可视化综合混淆矩阵（格式与参考图一致）
    plt.figure(figsize=(12, 10))
    
    # 绘制转置后的混淆矩阵：行(Y轴)=预测标签，列(X轴)=真实标签
    ax = sns.heatmap(confm_transposed, 
                     square=True, 
                     annot=True, 
                     fmt='d',
                     cbar=True,
                     linewidths=0.5,
                     linecolor='white',  # 白色网格线，类似参考图
                     cmap='Blues',
                     xticklabels=categories,
                     yticklabels=categories,
                     vmin=0,
                     vmax=confm_transposed.max() if confm_transposed.max() > 0 else 1,
                     cbar_kws={'label': '样本数量'})
    
    # 按照参考图格式：X轴=真实标签，Y轴=预测标签
    plt.xlabel('真实标签', size=14, fontweight='bold')
    plt.ylabel('预测标签', size=14, fontweight='bold')
    plt.title('综合混淆矩阵 (训练集+验证集+测试集，包含所有10个类别)', 
              size=16, fontweight='bold', pad=20)
    
    # 标签旋转45度，便于阅读
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 高亮对角线（正确分类）- 使用红色边框
    for i in range(len(categories)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    
    # 保存为1.png
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 综合混淆矩阵已保存到: {output_path}")
    print(f"   矩阵大小: {confm.shape} (10x10)")
    print(f"   包含的所有类别: {categories}")
    plt.close()


def evaluate_model(model, x_test, y_test, categories, save_dir, dataset_name='数据集'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        x_test: 测试集输入
        y_test: 测试集标签
        categories: 类别列表
        save_dir: 保存目录
        dataset_name: 数据集名称（用于显示）
    
    Returns:
        评估结果字典
    """
    # 预测
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 计算混淆矩阵（指定labels参数以确保包含所有10个类别）
    confm = confusion_matrix(y_true_classes, y_pred_classes, labels=list(range(len(categories))))
    
    # 打印分类报告
    # 使用labels参数指定所有10个类别，即使测试数据中可能没有包含所有类别
    print("\n分类报告:")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=categories, 
                                labels=list(range(len(categories))),
                                zero_division=0))
    
    # 可视化混淆矩阵（优化版本，确保所有10个类别都显示）
    plt.figure(figsize=(12, 10))
    
    # 确保混淆矩阵是10x10，即使某些类别在测试集中没有样本
    if confm.shape[0] < len(categories) or confm.shape[1] < len(categories):
        # 如果矩阵维度不足，创建完整的10x10矩阵
        full_confm = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(min(confm.shape[0], len(categories))):
            for j in range(min(confm.shape[1], len(categories))):
                full_confm[i, j] = confm[i, j]
        confm = full_confm
    
    # 计算每个类别的总数（用于归一化显示）
    row_sums = confm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # 避免除以0
    confm_normalized = confm.astype('float') / row_sums[:, np.newaxis]
    
    # 绘制热力图
    ax = sns.heatmap(confm, 
                     square=True, 
                     annot=True, 
                     fmt='d',  # 显示整数
                     cbar=True,
                     linewidths=0.5,
                     linecolor='gray',
                     cmap='Blues',  # 使用蓝色渐变，类似图二
                     xticklabels=categories,
                     yticklabels=categories,
                     vmin=0,
                     vmax=confm.max() if confm.max() > 0 else 1,
                     cbar_kws={'label': '样本数量'})
    
    # 设置标签和标题
    plt.xlabel('预测标签', size=14, fontweight='bold')
    plt.ylabel('真实标签', size=14, fontweight='bold')
    plt.title(f'{dataset_name}混淆矩阵 (包含所有10个类别)', size=16, fontweight='bold', pad=20)
    
    # 旋转标签以便更好地显示
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 高亮对角线（正确分类）
    for i in range(len(categories)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    
    conf_matrix_path = os.path.join(save_dir, 'text_classification_confusion_matrix.png')
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"混淆矩阵已保存到: {conf_matrix_path}")
    print(f"混淆矩阵大小: {confm.shape} (应等于 10x10)")
    print(f"矩阵中包含的所有类别: {categories}")
    plt.close()
    
    # 计算准确率
    accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confm.tolist(),
        'classification_report': classification_report(y_true_classes, y_pred_classes, 
                                                       target_names=categories,
                                                       labels=list(range(len(categories))),
                                                       output_dict=True,
                                                       zero_division=0)
    }


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置路径 - 基于脚本文件位置构建绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # nlp_deeplearn 目录
    base_dir = os.path.join(project_dir, 'data')
    save_dir = os.path.join(project_dir, 'tmp')
    
    train_dir = os.path.join(base_dir, 'cnews.train.txt')
    test_dir = os.path.join(base_dir, 'cnews.test.txt')
    val_dir = os.path.join(base_dir, 'cnews.val.txt')
    vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
    
    try:
        # 训练模型
        history, test_results = train_text_classification_model(
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
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"测试集准确率: {test_results['accuracy']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
