# Visualizing the Loss Landscape of Neural Nets


## Visualizing 2D loss contours

To plot the loss contours, we choose two random directions and normalize them in the same way as the 1D plotting.

```
mpirun -n 1 python plot_surface_mine.py --mpi --cuda --model vit_base_patch16_224 --x=-1:1:21 --y=-1:1:21 --model_file path_to_your_model --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --ngpu 6  --plot
```

![ResNet-56](doc/images/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5_train_loss_2dcontour.jpg)

Once a surface is generated and stored in a `.h5` file, we can produce and customize a contour plot using the script `plot_2D.py`.

```
python plot_2D.py --surf_file path_to_surf_file --surf_name train_loss
```
- `--surf_name` specifies the type of surface. The default choice is `train_loss`,
- `--vmin` and `--vmax` sets the range of values to be plotted.
- `--vlevel` sets the step of the contours.


## Visualizing 3D loss surface
`plot_2D.py` can make a basic 3D loss surface plot with `matplotlib`.
If you want a more detailed rendering that uses lighting to display details, you can render the loss surface with [ParaView](http://paraview.org).

![ResNet-56-noshort](doc/images/resnet56_noshort_small.jpg) ![ResNet-56](doc/images/resnet56_small.jpg)

To do this, you must
1. Convert the surface `.h5` file to a `.vtp` file.
```
python h52vtp.py --surf_file path_to_surf_file --surf_name train_loss --zmax  10 --log
```
   This will generate a [VTK](https://www.kitware.com/products/books/VTKUsersGuide.pdf) file containing the loss surface with max value 10 in the log scale.

2. Open the `.vtp` file with ParaView. In ParaView, open the `.vtp` file with the VTK reader. Click the eye icon in the `Pipeline Browser` to make the figure show up. You can drag the surface around, and change the colors in the `Properties` window.

3. If the surface appears extremely skinny and needle-like, you may need to adjust the "transforming" parameters in the left control panel.  Enter numbers larger than 1 in the "scale" fields to widen the plot.

4. Select `Save screenshot` in the File menu to save the image.
