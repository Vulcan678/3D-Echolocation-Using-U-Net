Using segmentation U-Net Network to mimic 3D echolocation, target detection, and range estimation, with overlapping returned echoes.

Simulating scenes of a bat transmitting an echo and receiving the echoes, in 2 ears, reflecting from targets in 3D space around the bat.
Scene limits:

| Variables           | Minimal       | Maximal |
| ------------------- |:-------------:| :------:|
| Number of targets   | 1             |      10 |
| Range of target (m) | 1             |       3 |
| Target size (m)     | 0.05          |      30 |

The scene simulation is based on the article:

Mazar, O., & Yovel, Y. (2020). A sensorimotor model shows why a spectral jamming avoidance response does not help bats deal with jamming. eLife, 9, e55539. https://doi.org/10.7554/eLife.55539

Due to echo length, small ranges, and large target numbers, many echoes overlap for each scene.

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/16a2fe88-4397-4efd-a33b-9cc1e5835634)

The U-Net Segmentation Network is used to get as input the received echoes in the time domain and perform segmentation of the number of echoes for each time sample. 
The original U-Net is used for image segmentation while here it is used for 1D signal segmentation.

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/6eb096cf-9594-42dc-b594-2ebd56ce7f4e)

The original U-Net is improved using Attention Blocks after the skip connections (AttU-Net):

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/1f8ff9bc-7827-4837-beb1-a74830ccbaa2)

As further improvement the attention blocks were replaced with modified Transformer Blocks (TransU-Net):

![Transformer](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/9b32ff78-ca33-4c50-a4df-1d9a0c29b9c2)

The signal and its segmentation label are presented in an example

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/f56cd14f-bc2d-45c4-ac0e-232a978ffdf0)

Due to large value differences by orders of magnitude, I performed a logarithmic normalization of the input signal

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/2046f580-d429-4bec-a4c4-ecddb77dd90f)

Training parameters:

| Parameter                | Value             |
| ------------------------ |:-----------------:|
| Dataset size             | 10000             |
| Training dataset size    | 6000              |
| Validation dataset size  | 1500              |
| Test dataset size        | 2500              |
| Maximal epochs           | 120               |
| Learning rate schedualer | CosineAnnealingLR |

To achieve the best training procedure and result I implemented hyperparameter optimization using HyperOpt and the variable training parameters were:
1. Batch size: 32 to 64 samples
2. Initial learning rate: 1e-8 to 1e-2
3. Weight decay: 0 to 0.1
4. $\beta_1$: 0.5 to 0.999
5. $\beta_2$: 0.9 to 0.9999
6. Minimal learning rate: 1e-11 to 1e-5
7. Loss function

The variable loss functions were:
1. Dice loss
2. Cross Entropy loss
3. Mean of Dice and Cross Entropy losses
4. IoU loss
5. Focal loss
6. Tversky loss ($\alpha=0.2, \beta=0.8$)

The output segmentation of the network is noisy so I applied an algorithm to filter changes in segmentations.
The algorithm output is a segmentation based on pairs of segmentation value jump and segmentation value drop where the distance between them is equal to echo length, with an allowed error threshold.

For performance evaluation classic target detection and range estimation metrics are used:
1. Mean absolute range error (MAE) =
$$\left( 1/N \right) \sum_{n=1}^N \Delta r_n $$
2. Miss detection rate (MD rate) =
$$\left( 1/N \right) \sum_{n=1}^N \left[ num \left( r_n^{true} \right) - num \left(r_n^{est} \right) \right]*I_{num \left(r_n^{true} \right) > num \left(r_n^{est} \right)} $$
3. False alarm rate (FA rate) =
$$\left( 1/N \right) \sum_{n=1}^N \left[ num \left( r_n^{est} \right) - num \left(r_n^{true} \right) \right]*I_{num \left(r_n^{est} \right) > num \left(r_n^{true} \right)} $$

From all the models trained and evaluated on the validation set from the hyperparameter optimization the models on the Pareto front are found by the metrics: MAE, MD rate, FA rate and Dice loss. From the Pareto front, the selected best model is by the minimal sum of ranking of the models' performance metrics relative to the other Pareto front models.

The result of different network architectures:

| Metric        | U-Net      | AttU-Net | TransU-Net |
| ------------- |:----------:|:--------:|:----------:|
| MAE (cm)      | 1.51       | **1.48** | 1.61       |
| MD Rate       | 0.0588     | 0.0591   | **0.0568** |
| FA Rate       | **0.0347** | 0.0365   | 0.0379     |
| Dice Loss (%) | 96.23      | 97.88    | **98.33**  |
