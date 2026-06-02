# 可视化 `test_sv` 跟踪结果

该脚本用于把 `seqtrack` 的预测框结果叠加到原始视频帧上，并导出为 `mp4`。

- 脚本位置：`tracking/visualize_sv_results.py`
- 默认输入结果目录：`output/test/tracking_results/seqtrack/seqtrack_b384_3d`
- 默认输出目录：`output/test/visualizations/seqtrack_b384_3d`

## 快速使用

在 `seqtrack3d` 根目录执行：

```bash
source /home/silas/miniconda3/etc/profile.d/conda.sh
conda activate seqtrackv2
python tracking/visualize_sv_results.py --sequence 01_000000 --draw_gt
```

## 常用参数

- `--sequence`：只渲染单个序列（例如 `01_000000`），不传则渲染所有结果。
- `--draw_gt`：叠加真值框（绿色）。预测框始终显示为红色。
- `--max_frames`：仅渲染前 N 帧，用于快速预览。
- `--fps`：输出视频帧率。
- `--dataset_root`、`--results_dir`、`--output_dir`：可覆盖默认路径。

## 输出文件

每个序列生成一个视频：

`<output_dir>/<sequence>.mp4`

例如：

`output/test/visualizations/seqtrack_b384_3d/01_000000.mp4`
