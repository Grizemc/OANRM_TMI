# OANRM 点云对应训练与 无监督微调 后训练说明

本项目用于带颜色信息的点云对应预测，模型输出源点云到目标点云的预测坐标，以及两帧之间重叠区域的掩码。当前推荐流程为：

1. 使用服务器上的 **mix 数据集**训练基础模型。
2. 使用训练得到的 `best_model.t7`，在 Hamlyn 数据上执行逐样本无监督后训练（post-training）与结果导出。

本文档对应的入口文件如下：

| 用途 | 入口文件 | 配置文件 |
| --- | --- | --- |
| 基础模型训练 | `mask_main_small_normalizeOfSource_caixiang_mix.py` | `config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml` |
| Hamlyn 75 后训练/测试 | `PostTrain_Hamlyn_75.py` | `config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml` |

> 所有命令均应在项目根目录（即本 `README.md` 所在目录）运行。代码使用了 `cp` 命令、CUDA 扩展和 Linux 风格的数据路径，推荐在 Linux GPU 服务器上运行。

## 1. 目录说明

```text
Paconv_730/
├── mask_main_small_normalizeOfSource_caixiang_mix.py   # mix 数据集训练入口
├── PostTrain_Hamlyn_75.py                              # Hamlyn 75 后训练/测试入口
├── config/
│   └── Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
│                                                       # 推荐使用的训练与测试配置
├── util/
│   ├── data.py                                        # 数据集读取逻辑
│   └── util.py                                        # YAML 配置读取逻辑
├── model/                                             # PAConv、骨干网络和损失函数
├── lib/
│   ├── pointops/                                      # 必须编译的 CUDA 点云算子
│   └── ChamferDistancePytorch/chamfer3D/        # 测试所需 
Chamfer Distance CUDA 算子
├── checkpoints/
│   └── Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/
│       └── saved_model/best_model.t7                  # 训练最佳权重 / 测试加载权重
└── README.md
```

## 2. 环境与依赖

本项目直接使用服务器 **3090** 上已配置好的 `szmpa` conda 虚拟环境。运行前激活该环境：

```bash
conda activate szmpa
```

不要在本项目目录中重新创建环境或安装其他版本的 PyTorch、CUDA；训练和测试均依赖该服务器环境中已匹配的 PyTorch、CUDA、Open3D、TensorBoardX、SciPy 与编译工具链。

## 3. 配置文件放置位置与修改方法

推荐配置文件必须放在项目根目录下的 `config/` 目录中：

```text
config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
```

训练命令应显式传入该相对路径。训练脚本内部的默认路径是旧服务器路径 `/home/szm/Paconv_730/...`，迁移服务器后不能依赖默认值。

```bash
python mask_main_small_normalizeOfSource_caixiang_mix.py \
  --config config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
```

测试脚本默认也指向同名配置；为避免当前目录错误，仍建议显式传参：

```bash
python PostTrain_Hamlyn_75.py \
  --config config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
```

要创建新实验，请复制该 YAML 至 `config/` 下的新文件，例如：

```bash
cp config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml \
  config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix_myexp.yaml
```

然后至少修改 `Model.exp_name`，防止新旧实验写入同一 `checkpoints/` 目录。YAML 中的顶层分组（`DATA`、`Model`、`Train`、`TEST`）会在读取时合并为同一组参数，因此代码中通过 `args.data_dir`、`args.exp_name` 等名称访问。

## 4. mix 数据集配置

当前推荐配置为：

```yaml
DATA:
  data_dir: /big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_mix
  num_points: 8192
  cuda: True
  normalize: True

Model:
  exp_name: Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix
  model_type: Base_flow
  dataset: normal

Train:
  train_batch: 32
  epochs: 150

TEST:
  test_batch: 16
```

其中 `dataset: normal` 是加载器分支名称，不表示普通数据集；在这个实验中，真正决定使用 mix 数据集的是 `DATA.data_dir` 指向服务器上的 mix 数据根目录。

### mix 数据集目录结构

训练加载器 `MaskMICCAIMutualNormalized` 会直接读取以下文件：

```text
/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_mix/
├── train/
│   ├── *.npz
│   └── ...
└── test/
    ├── *.npz
    └── ...
```

每个 `.npz` 至少应包含下列键。每帧点数必须不小于 `num_points`（默认 8192），因为加载器按无放回随机采样。

| NPZ 键 | 含义 | 常见形状 |
| --- | --- | --- |
| `mask_point1` | 源帧点云坐标 | `[N1, 3]` |
| `mask_color1` | 源帧点颜色/特征 | `[N1, 3]` |
| `mask_point2` | 目标帧点云坐标 | `[N2, 3]` |
| `mask_color2` | 目标帧点颜色/特征 | `[N2, 3]` |
| `mask_gt1` | 源帧有效对应/重叠掩码 | `[N1]` |
| `mask_gt2` | 目标帧有效对应/重叠掩码 | `[N2]` |
| `mask_gt_pc` | 源帧点的目标对应真值坐标 | `[N1, 3]` |

迁移到其他服务器时，修改 YAML 的唯一数据集路径项即可：

```yaml
DATA:
  data_dir: /your/server/path/M8ICCAI_8192_Mask_19055_new_mutual_mix
```

不要把服务器绝对路径写进训练命令；集中维护在 YAML 中，便于复现实验。

## 5. 选择 GPU

两个入口脚本内部写死了 GPU 可见卡：

| 脚本 | 当前位置 | 默认值 |
| --- | --- | --- |
| `mask_main_small_normalizeOfSource_caixiang_mix.py` | 文件开头的 `CUDA_VISIBLE_DEVICES` | `6` |
| `PostTrain_Hamlyn_75.py` | 文件开头的 `CUDA_VISIBLE_DEVICES` | `5` |

请在运行前将对应脚本中的值改为本服务器可用 GPU 编号，例如：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

仅在 shell 前加 `CUDA_VISIBLE_DEVICES=0` 不足以覆盖这个设置，因为脚本启动后会再次赋值。训练脚本使用 `torch.nn.DataParallel`，可见的多张 GPU 会参与训练；只设置一张可见卡时则以单卡方式运行。

## 6. 训练 mix 基础模型

确认以下条件后启动训练：

1. `config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml` 中的 `data_dir` 可访问。
2. `data_dir/train` 与 `data_dir/test` 中已有可读取的 `.npz` 文件。
3. 两个 CUDA 扩展已完成编译。
4. 训练脚本中的 GPU 编号已调整。

### 运行前的代码一致性检查

当前仓库中的 `util/data.py` 与训练入口存在一处已知的返回值数量不一致：

- `MaskMICCAIMutualNormalized.__getitem__()` 返回 10 项数据，其中最后两项为 `relax_ratio` 与 `mask_point1_source`。
- `mask_main_small_normalizeOfSource_caixiang_mix.py` 中的 `train_one_epoch()` 和 `test_one_epoch()` 目前各只解包 9 项。

因此，直接运行当前版本可能报错：`ValueError: too many values to unpack`。开始正式训练前，应将这两个位置的解包语句从：

```python
mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_gt_pc, gt_pc, _ = data
```

改为：

```python
mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_gt_pc, gt_pc, _, _ = data
```

这仅丢弃当前训练流程未使用的最后两项，其他训练逻辑保持不变。若后续替换了 `util/data.py`，请再次核对数据集返回值和训练循环解包数量是否一致。

训练命令：

```bash
python mask_main_small_normalizeOfSource_caixiang_mix.py \
  --config config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
```

训练脚本会在以下路径创建日志、TensorBoard 事件文件、运行时备份代码和最佳模型：

```text
checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/
├── train.log
├── train/                              # TensorBoard 事件文件
├── pythonfile/                         # 启动训练时备份的 model/ 与 util/
└── saved_model/
    └── best_model.t7                   # 按测试集 1 mm 准确率保存的最佳权重
```

查看训练曲线：

```bash
tensorboard --logdir checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/train --port 6006
```

训练日志会记录点对应误差、1/2/3/5/10 mm 阈值准确率、掩码阈值准确率，以及各损失项。`best_model.t7` 会在测试集的 1 mm 准确率提升时覆盖更新。

### 训练参数重点

| 配置项 | 当前值 | 作用 |
| --- | ---: | --- |
| `num_points` | 8192 | 每个点云采样点数 |
| `train_batch` | 32 | 训练批大小，显存不足时首先降低此值 |
| `test_batch` | 16 | 验证批大小 |
| `epochs` | 150 | 训练轮数 |
| `lr` | 0.001 | 初始学习率 |
| `learning_ratedeacy` | `cosine` | 余弦学习率调度 |
| `color_aug` | `True` | 训练时对颜色使用扰动增强 |
| `loss_type` | `one_loss_base` | 当前损失函数选择 |
| `gt_factor` / `smooth_factor` / `mask_factor` | 8 / 10 / 4 | 各损失项权重 |

显存不足时，优先下调 `train_batch`，其次才考虑下调 `num_points`；若修改 `num_points`，训练集和测试集应保持一致，并确认网络相关参数与显存容量相适配。

## 7. Hamlyn 75 后训练与测试

`PostTrain_Hamlyn_75.py` 不进行基础模型的常规验证，而是先加载 mix 训练权重，再对 Hamlyn 样本进行逐个无监督后训练。它依赖 Chamfer Distance、点云法向估计以及 FPFH 伪对应。

### 7.1 权重位置

测试脚本固定从下列路径加载权重：

```text
checkpoints/<exp_name>/saved_model/best_model.t7
```

对于默认 mix 配置，完整路径为：

```text
checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/saved_model/best_model.t7
```

运行测试前必须确认该文件存在。若使用其他实验权重，应在 YAML 中将 `Model.exp_name` 改为该权重所在目录名，或将权重放到上述约定位置。

### 7.2 Hamlyn数据路径

测试数据路径目前直接写在 `PostTrain_Hamlyn_75.py` 的主程序中，而不在 YAML 中，
后续要测试不同重叠数据集的效果，直接在该 `PostTrain_Hamlyn_75.py` 文件中修改以下路径：

```python
root = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/test1"
fpfh_path = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/fpft_file1"
```

若要测试其余重叠数据集，请修改这两个路径值为实际的 Hamlyn 各重叠数据集目录和 FPFH 目录。(其余重叠率数据集在目录"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_xxx"下)

Hamlyn 样本 `.npz` 应包含与训练集相同的基础字段：`mask_point1`、`mask_color1`、`mask_point2`、`mask_color2`、`mask_gt1`、`mask_gt2`、`mask_gt_pc`。每个 FPFH `.npz` 必须包含：

```text
matches_list0    # [K, 2]，每行分别是源点索引与目标点索引
```

### 7.3 启动测试

确认 GPU 编号、权重路径、Hamlyn 数据路径、FPFH 数据路径后运行：

```bash
python PostTrain_Hamlyn_75.py \
  --config config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml
```

当前脚本使用的后训练初始参数写在主程序内：`pt_epoch=8`、`post_train_lr=0.001`、`gt_factor=2.0`、`smooth_factor=1.0`、`mask_truncation=0.8`，并启用 `need_fpfh=True`。这些不是 YAML 配置项；若需要调整，请修改测试脚本末尾的相应赋值。

### 7.4 测试输出与断点续跑

输出位置：

```text
checkpoints/<exp_name>/
└── fpfh_Post_Train_Hamlyn_no_rotation_75_datiao_1017/
    ├── post_train.log
    └── npz_result/
        └── post_train_sample_num_<index>_best.npz
```

每个导出的结果文件通常包括输入点云、颜色、预测坐标 `pred_xyz`、预测掩码 `pred_mask1`/`pred_mask2`、真值掩码及误差。脚本启动时会检查对应的输出文件是否存在；存在则跳过该样本，因此可在任务中断后直接使用同一命令继续运行。

如需重新处理某个样本，应先备份或删除该样本对应的 `post_train_sample_num_<index>_best.npz` 文件。若需要重新处理全部样本，请仅清理该实验目录下的 `npz_result/`，不要误删基础模型的 `saved_model/best_model.t7`。

## 8. Hamlyn 结果评估与 TDDP 后处理

评估脚本位于：

```text
ReadData/FinalEval/FinalReadHamlynResultMore.py
```

该脚本以 Hamlyn 后训练输出目录为输入，即包含 `npz_result/` 的目录。例如：

```text
checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/
fpfh_Post_Train_Hamlyn_no_rotation_75_datiao_1017/
```

三个核心函数的处理顺序与职责如下：

```text
无监督后训练结果（npz_result/*.npz）
        |
        +--> DirectCalReadHanlynResult(root_path)
        |      直接评估无监督微调后的结果
        |
        +--> GaussianPostMostProcessHamlynResult(root_path)
               执行 TDDP 后处理，并保存后处理结果
                       |
                       +--> CalMostGaussinaPostProcessHamlynResult(root_path)
                              评估 TDDP 后处理后的结果
```

| 函数 | 用途 |
| --- | --- |
| `DirectCalReadHanlynResult(root_path)` | 对无监督微调后直接导出的 `npz_result` 进行评估。 |
| `GaussianPostMostProcessHamlynResult(root_path)` | 对无监督微调结果执行 TDDP 后处理，并将后处理结果保存到该实验目录。 |
| `CalMostGaussinaPostProcessHamlynResult(root_path)` | 对 TDDP 后处理生成的结果进行评估。 |

调用时，将 `root_path` 设为实验结果目录，而不是 `npz_result/` 子目录。建议严格按上述顺序执行：先得到直接评估结果，再完成 TDDP 后处理，最后评估后处理结果。函数名称中的 `Gaussina` 为脚本现有拼写，调用时应保持一致。

## 9. 常见问题排查

| 现象 | 检查与处理 |
| --- | --- |
| `... is not a yaml file` 或找不到配置 | 从项目根目录运行，并显式传入 `--config config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml`。 |
| `No module named pointops_cuda` | 在当前 conda 环境执行 `cd lib/pointops && python setup.py install`。 |
| Chamfer 模块导入失败 | 执行 `cd lib/ChamferDistancePytorch/chamfer3D && python setup.py install`，并确认 PyTorch/CUDA/编译器兼容。 |
| `No model` 或找不到 `best_model.t7` | 先完成训练，或检查 YAML 中 `exp_name` 是否与权重目录相同。 |
| 找不到数据或数据集长度为 0 | 检查 YAML 中的 `data_dir` 及其 `train/`、`test/` 子目录；Hamlyn 测试还要检查脚本内 `root` 与 `fpfh_path`。 |
| `Cannot take a larger sample than population` | 某个 `.npz` 的点数小于 `num_points=8192`；补足数据或降低 YAML 中的 `num_points`。 |
| CUDA out of memory | 降低 `train_batch` 或 `test_batch`；确认没有其他进程占用当前 GPU。 |
| 运行到了错误 GPU | 两个入口文件内部都会设置 `CUDA_VISIBLE_DEVICES`，请直接修改脚本中的编号。 |
| `ValueError: too many values to unpack` | 按“运行前的代码一致性检查”一节，将训练与验证循环的解包改为 10 项。 |
| Hamlyn 测试结果错位 | 核对数据 `.npz` 与 FPFH `.npz` 的数量、命名和排序是否一一对应。 |

## 10. 复现实验前检查清单

- [ ] 已激活正确的 conda 环境，并且 `torch.cuda.is_available()` 为 `True`。
- [ ] `pointops` 与 `chamfer3D` CUDA 扩展已编译。
- [ ] YAML 文件位于 `config/`，且训练命令通过 `--config` 显式指定。
- [ ] `DATA.data_dir` 指向服务器的 mix 数据集根目录，并含有 `train/`、`test/`。
- [ ] 已确认训练/测试脚本内的 GPU 编号。
- [ ] 测试前存在 `checkpoints/<exp_name>/saved_model/best_model.t7`。
- [ ] Hamlyn 测试前已修改 `PostTrain_Hamlyn_75.py` 中的 `root` 与 `fpfh_path`。
- [ ] Hamlyn 数据与 FPFH 文件数量、名称和顺序一致。
