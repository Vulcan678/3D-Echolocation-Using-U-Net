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

The signal and its segmentation label are presented in an example

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/f56cd14f-bc2d-45c4-ac0e-232a978ffdf0)

Due to large value differences by orders of magnitude I performed and logarithmic normalization of the input signal

![image](https://github.com/Vulcan678/3D-Echolocation-Using-U-Net/assets/153300908/2046f580-d429-4bec-a4c4-ecddb77dd90f)

The output segmentation of the network is noisy so I applied an algorithm to filter changes in segmentations.
The algorithm output is a segmentation based on pairs of segmentation value jump and segmentation value drop where the distance between them is equal to echo length, with allowed error threshold.
