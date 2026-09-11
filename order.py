#恢复训练   python train.py --resume output/26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04/checkpoint/latest

"""
测试demo 
  python test_demo.py --exp_path 26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04 --task_name IVF --dataset_name VTMOT-demo --batch_size 1

  python test_demo.py \
    --exp_path 26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04 \
    --task_name IVF \
    --dataset_name VTMOT-demo \
    --batch_size 4 \
    --num_workers 8
"""

"""测试
python test.py \
  --exp_path 26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04\
  --ckpt_path latest \
  --task_name IVF \
  --dataset_name VTMOT

python test.py \
 --exp_path 26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04 \
 --ckpt_path latest \
 --task_name IVF \
 --dataset_name VTMOT \
 --batch_size 4 \
 --num_workers 8
"""

"""损失曲线
python plot_loss.py --log output/26_03_16-15_50_38-IVF-crop128-bs32_4-coef1_5_4_0-lr1e-04/logging.log
"""