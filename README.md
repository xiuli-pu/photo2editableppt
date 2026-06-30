# photo2editableppt

> 可编辑 OCR 文字的颜色不再尽量估计原文字颜色，而是根据文字框下方的局部背景自动选择高对比颜色。因为课堂投影照片中原文字颜色估计很容易被背景、曝光、投影色偏影响。

默认规则：

- 绿色、蓝色、紫色、红色、深色、饱和色背景：使用白色可编辑文字；
- 白色、浅灰、浅黄等浅色背景：使用黑色可编辑文字；
- 目的不是完全还原原始 PPT 字体颜色，而是保证可编辑文字在背景中清楚、突出。

## 运行

```bash
解压缩附带的代码项目压缩包，拍摄的ppt数据集图片在data/raw_photos文件夹中
cd photo2editableppt
conda create env
conda activate env
python -m pip install -r requirements.txt
python run.py

```

输出：

```text
output/01_reference_rectified_image_ppt.pptx
output/02_rebuilt_editable_ppt.pptx
在我的解压文件中也可以直接看到两个生成的ppt文件
```

## OCR 环境建议

Windows + Python 3.8 建议：

```bash
python -m pip install paddlepaddle==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install lmdb==1.4.1 --only-binary=:all: -i https://pypi.org/simple
python -m pip install paddleocr==2.7.3
```

如果遇到 OpenMP 报错，本项目 `run.py` 已经在最开头设置：

```python
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

## 关键配置

在 `config.yaml` 中：

```yaml
rebuild:
  background_mode: clean_overlay
  text_color_policy: contrast_with_background
  contrast_text_mode: high_contrast_bw
```


