## 下载预训练模型

```bash
cd pretrain
python download.py
```

## 数据准备

1. 访问数据集页面：HuggingFace上的[CryCeleb2023](https://huggingface.co/datasets/Ubenwa/CryCeleb2023/tree/main)
2. 手动下载以下文件：
   - `audio.zip`（包含所有音频文件）
   - `dev_pairs.csv`
   - `metadata.csv`
   - `test_pairs.csv`
3. 在项目文件夹中创建`data`目录
4. 将所有下载的文件放入`data`目录
5. 解压音频文件：
```bash
unzip data/audio.zip -d data/
rm data/audio.zip
```

## 运行命令

### 训练模式

```bash
python main.py --is_train True --model whisper
```

### 测试模式

```bash
python main.py --is_train False --model whisper --seed 2024 --max_audio none
```
or 
```bash
python main.py --is_train False --model whisper --seed 2024 --max_audio $((600 * 160+240))
```

##   参考代码

https://github.com/Ubenwa/cryceleb2023

https://github.com/conti748/cryceleb2023