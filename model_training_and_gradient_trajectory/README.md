## How To Replicate The Results

### Training Models

```
python -m My_CIFAR_TIMM --model vit_base_patch16_224 --origin_params 'patch_embed.proj.bias' --save_log 0 --clipping_mode BK-MixOpt --epochs 2 --cifar_data CIFAR100 --device cuda:0 --bs 2000 --exp_name BS_2000
```

More arguments see the `My_CIFAR_TIMM.py` file for details.



### GPU Occupation

```plaintext
python GPU_Occupation.py
```

The results will be saved in the `result` folder. 

Use `ViT_CIFAR_GPU_result\Compare_DP_n_nonDP\plot.ipynb` to plot.



### Gradient Trajectory

Train the models and save them first.

```
python -m Trajectory_save_model --model vit_base_patch16_224 --origin_params 'patch_embed.proj.bias' --save_log 0 --clipping_mode BK-MixOpt --epochs 10 --cifar_data CIFAR100 --lr 4e-4 --device cuda:1 --exp_name DP_training
```

Then use the `Trajectory_DP/nonDP.py` to get the gradients.

```
python  Trajectory_DP/nonDP.py
```

Finally, in the `gradient` folder, use the `.ipynb` file to plot.
