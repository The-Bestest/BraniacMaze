# BraniacMaze

A Brain Computer Interface algorithm, which classifies brain signal into 2 Motor Imagery classes + REST class. It is implemented from scratch in Python and scipy for educational purposes.

The implementation follows rather closely (but not absolutely) this [Reyhani-Masoleh](https://arxiv.org/pdf/1912.04828.pdf) pre-print paper (OpenAccess on arXiv).

The dataset comes from the [BCI Competition IV: Dataset 1](http://www.bbci.de/competition/iv/desc_1.html). Due to the licensing issues, you need to download the data yourself and agree with their terms of use.

In order to run the project you will need to download the 100Hz dataset and save the following files in the project.

Calibration files:
* `calibration/BCICIV_calib_ds1a.mat`
* `calibration/BCICIV_calib_ds1b.mat`
* `calibration/BCICIV_calib_ds1c.mat`
* `calibration/BCICIV_calib_ds1d.mat`
* `calibration/BCICIV_calib_ds1e.mat`
* `calibration/BCICIV_calib_ds1f.mat`
* `calibration/BCICIV_calib_ds1g.mat`

Evaluation files:
* `evaluation/BCICIV_eval_ds1a.mat`
* `evaluation/BCICIV_eval_ds1b.mat`
* `evaluation/BCICIV_eval_ds1c.mat`
* `evaluation/BCICIV_eval_ds1d.mat`
* `evaluation/BCICIV_eval_ds1e.mat`
* `evaluation/BCICIV_eval_ds1f.mat`
* `evaluation/BCICIV_eval_ds1g.mat`

You are free to use and distribute the Python code though as you see fit - but without any guarantees implied.

Graphical representation of the algorithm can be found on Miro: 

![The current design of BCI](https://github.com/The-Bestest/BraniacMaze/blob/master/BCI_design.jpg)

[Link to Miro](https://miro.com/app/board/o9J_lP_hATc=/)


