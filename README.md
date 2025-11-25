<div align="center">
<h1>
  星辰语音大模型-超多方言ASR
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0" target="_blank">Hugging Face</a>️ • 🤖 <a href="https://www.modelscope.cn/models/TeleAI/TeleSpeech-ASR1.0/summary" target="_blank">ModelScope</a> • 🐾 <a href="https://gitee.com/Tele-AI/TeleSpeech-ASR" target="_blank">gitee</a>
</p>

# 目录
- [目录](#目录)
- [模型开源](#模型开源)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
  - [特征提取](#特征提取)
  - [字典准备](#字典准备)
- [微调模型推理流程示例\*](#微调模型推理流程示例)
- [推理与解码](#推理与解码)
- [开源数据集结果](#开源数据集结果)
- [声明与协议](#声明与协议)
  - [声明](#声明)
  - [协议](#协议)

# 模型开源

星辰超多方言语音识别大模型v1.0，由30万小时无标注多方言语音数据进行预训练，并利用内部30种有标注数据进行微调，打破单一模型只能识别特定单一方言的困境，可支持理解粤语、上海话、四川话、温州话等30种方言


本次开源三个模型：两个30万小时无标注语音预训练模型和一个KeSpeech数据集8种方言微调模型。发布版本和下载链接见下表

| 模型版本             | 参数量  | 下载链接     | 字典  | 备注 |
|---------------------|-------|---------------------|-------|-------|
| pretrain_base | 0.09 B | [TeleSpeech-ASR1.0-base](https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0/blob/main/base.pt)  | ✗ | 30万小时无标注语音预训练模型 |
| pretrain_large | 0.3 B | [TeleSpeech-ASR1.0-large](https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0/blob/main/large.pt)  | ✗ | 30万小时无标注语音预训练模型 |
| finetune_large_kespeech | 0.3 B | [TeleSpeech-ASR1.0-large-kespeech](https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0/blob/main/finetune_large_kespeech.pt) | [dict.char7531.txt](https://huggingface.co/Tele-AI/TeleSpeech-ASR1.0/blob/main/dict.chr7531.txt) | 基于pretrain_large，采用KeSpeech数据集[8种方言](#KeSpeech各方言上结果)微调训练|

* finetune模型为已经在特定数据集微调过的模型，可直接使用
* pretrain模型为无监督预训练模型，**无法直接进行ASR任务**，需要用少量标注数据进行有监督训练后使用。相比于直接训练的方言识别模型，基于预训练模型可以利用更少的有标注数据获得更好的方言识别性能。本仓库现聚焦于推理解码链路，如需完整训练流程请参考官方fairseq或其他开源方案

# 环境配置

* PyTorch version >= 1.13.0
* Python version >= 3.8
* 数据准备需要使用kaldi，请确保已正确安装：https://github.com/kaldi-asr/kaldi
  * 若已有提好的特征，程序运行时可以使用wenet开源框架中kaldi_io.py实现的方法替换kaldiio.load_mat，从而无需安装kaldi

<a id="fairseq安装"></a>
* 本仓库已在`fairseq/`目录内自带精简后的fairseq源码（v0.12.2），无需额外编译或安装即可用于推理脚本；为避免C/CUDA扩展编译，已移除原版fairseq中的数据集/任务定义、训练器、优化器、学习率调度器以及所有损失函数等训练相关逻辑。如果需要完整训练链路，请改用官方fairseq安装包；若希望使用系统已安装的fairseq，可自行调整`PATH`/`PYTHONPATH`。

* 推理无关的wenet表征训练等模块已从仓库移除，仅保留必要的解码脚本与依赖

* 安装推理解码所需依赖
```shell script
$ pip install -r requirements.txt
```

* 若只需要运行解码脚本，可以不安装完整的requirements.txt，只需保证kaldiio, timm, editdistance, soundfile已正确安装
```shell script
$ pip install kaldiio timm editdistance soundfile
```


# 数据准备
## 特征提取
<a id="特征提取"></a>

* 模型输入为从16K采样率音频中提取的40维mfcc特征，**非原始音频**
* 利用kaldi提取40维mfcc特征，运行脚本参考`prepare_kaldi_feats.sh`
  * 可将运行脚本`prepare_kaldi_feats.sh`与参数设置`mfcc_hires.conf`置于kaldi任一egs目录下（与cmd.sh等脚本平级，例如/path/to/kaldi/egs/aishell/s5/prepare_kaldi_feats.sh），运行`prepare_kaldi_feats.sh`即可
* 为各数据集准备训练用文件`data.list`，可参考`make_datalist.py`，以`\t`分隔：
```
$ cat train/data.list
utt:X0000000000_100638174_S00037	feat:/data/raw_nnaudio.test.1.ark:2983479385	feat_shape:363,40	text:不惜在这种试验中毁灭包括自己在内的一切	token:不 惜 在 这 种 试 验 中 毁 灭 包 括 自 己 在 内 的 一 切	tokenid:[TOKENID]	token_shape:19,5537
utt:X0000000001_100849618_S00006	feat:/data/raw_nnaudio.test.1.ark:2984296665	feat_shape:345,40	text:在他们收到足够建立大统一模型的数据后	token:在 他 们 收 到 足 够 建 立 大 统 一 模 型 的 数 据 后	tokenid:[TOKENID]	token_shape:18,5537
...
```

## 字典准备

* 微调阶段，需要准备fairseq格式的 `dict.${label}.txt`，`${label}`为建模单元类型，如ltr, bpe等。以`dict.ltr.txt`为例：
```
是 2
好 3
...
```

# 微调模型推理流程示例*
1. [fairseq环境准备](#fairseq安装)，运行`data2vec_dialect/path.sh`会自动将仓库内置的`fairseq/`加入`PYTHONPATH`
2. 利用kaldi提取音频特征，准备data.list格式文件，参考[特征提取](#特征提取)，并命名为以 .tsv 结尾的文件
   * data.list中`text`、`token`是为了微调和统计CER使用，若只想拿到解码结果，data.list中的`text`、`token`只需保证有内容即可 
3. 进入data2vec_dialect目录，并修改`run_scripts/decode.sh`文件，参考[推理与解码](#推理与解码)
4. 在data2vec_dialect路径下，执行`run_scripts/decode.sh`

*仅经过微调后的finetune模型支持直接推理，无监督预训练模型`pretrain_base`和`pretrain_large`需要先在标注数据上训练后，再进行推理，相关训练链路不再随仓库提供。

<a id="推理与解码"></a>

# 推理与解码
* 修改`run_scripts/decode.sh`中的模型路径、测试数据路径等
  * `dataset.gen_subset`为测试数据路径下 .tsv 文件的名称，可配置多个
* 在data2vec_dialect路径下执行
    ```shell script
    $ bash run_scripts/decode.sh
    ```
# 开源数据集结果
* 我们选择了多个开源中文数据集进行验证，以测试集上的字错误率 (Character Error Rate, CER) 结果作为衡量标准
* 在Aishell-1上我们选择其Train集作为有监督数据进行训练，在Test集上统计CER
* 在WenetSpeech上，我们分别使用100小时训练集Train_s和1000小时训练集Train_m分别作为有监督数据进行训练，在Test_Meeting测试集上统计CER
* Babel为NIST（美国国家标准与技术研究院）举办的低资源粤语电话识别任务数据集，我们使用其提供的训练集与测试集统计CER
* KeSpeech为中文多方言测试集，我们使用1396小时训练集作为有监督数据进行训练，选择提供的Test测试集统计CER

|  模型版本         | Aishell-1 (%)| WenetSpeech* (%)| Babel (%) | KeSpeech (%) |
| ----------| -------- | ------- | ---- | ---- |
| pretrain_base | 4.7  | 18.3 / 16.4 | 22.1  | 10.9 |
| pretrain_large | 4.0 | 14.3 / 13.0 | 19.1  | 8.1 |

*WenetSpeech中的结果为分别使用 `train_s/train_m`训练后，在Test_Meeting上的CER

<a id="KeSpeech各方言上结果"></a>
KeSpeech各方言上结果（CER%）
|  模型版本 | 普通话 | 北京 | 西南 | 中原 | 东北 | 兰银 | 江淮 | 冀鲁 | 胶辽 |
| ---------| ------ | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- |
| pretrain_large | 4.61 | 8.23 | 8.74 | 7.62 | 7.89 | 9.72 | 12.89 | 8.91 | 9.30 |

# 声明与协议
## 声明
我们在此声明，不要使用TeleSpeech模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将TeleSpeech模型用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用TeleSpeech开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

## 协议
社区使用TeleSpeech模型需要遵循《[TeleSpeech模型社区许可协议](./TeleSpeech模型社区许可协议.pdf)》。TeleSpeech模型支持商业用途，如果您计划将TeleSpeech模型或其衍生品用于商业目的，您需要通过以下联系邮箱 tele_ai@chinatelecom.cn，提交《TeleSpeech模型社区许可协议》要求的申请材料。审核通过后，将特此授予您一个非排他性、全球性、不可转让、不可再许可、可撤销的商用版权许可。

---