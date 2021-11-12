# Contrastive Learning Inverts the Data Generating Process [ICML 2021]
Official code to reproduce the results and data presented in the paper [Contrastive Learning Inverts the Data Generating Process](https://brendel-group.github.io/cl-ica/).

<p align="center">
  <img src="https://brendel-group.github.io/cl-ica/img/overview_compressed.svg" alt="3DIdent dataset example images" />
</p>

## Experiments 
To reproduce the disentanglement results for the MLP mixing, use the [main_mlp.py](main_mlp.py) script. For the experiments on KITTI Masks use the [main_kitti.py](main_kitti.py) script. For those on 3DIdent, use [main_3dident.py](main_3dident.py).

### MLP Mixing

```bash
> python main_mlp.py --help
usage: main_mlp.py
       [-h] [--sphere-r SPHERE_R] [--box-min BOX_MIN] [--box-max BOX_MAX]
       [--sphere-norm] [--box-norm] [--only-supervised] [--only-unsupervised]
       [--more-unsupervised MORE_UNSUPERVISED] [--save-dir SAVE_DIR]
       [--num-eval-batches NUM_EVAL_BATCHES] [--rej-mult REJ_MULT]
       [--seed SEED] [--act-fct ACT_FCT] [--c-param C_PARAM]
       [--m-param M_PARAM] [--tau TAU] [--n-mixing-layer N_MIXING_LAYER]
       [--n N] [--space-type {box,sphere,unbounded}] [--m-p M_P] [--c-p C_P]
       [--lr LR] [--p P] [--batch-size BATCH_SIZE] [--n-log-steps N_LOG_STEPS]
       [--n-steps N_STEPS] [--resume-training]

Disentanglement with InfoNCE/Contrastive Learning - MLP Mixing

optional arguments:
  -h, --help            show this help message and exit
  --sphere-r SPHERE_R
  --box-min BOX_MIN     For box normalization only. Minimal value of box.
  --box-max BOX_MAX     For box normalization only. Maximal value of box.
  --sphere-norm         Normalize output to a sphere.
  --box-norm            Normalize output to a box.
  --only-supervised     Only train supervised model.
  --only-unsupervised   Only train unsupervised model.
  --more-unsupervised MORE_UNSUPERVISED
                        How many more steps to do for unsupervised compared to
                        supervised training.
  --save-dir SAVE_DIR
  --num-eval-batches NUM_EVAL_BATCHES
                        Number of batches to average evaluation performance at
                        the end.
  --rej-mult REJ_MULT   Memory/CPU trade-off factor for rejection resampling.
  --seed SEED
  --act-fct ACT_FCT     Activation function in mixing network g.
  --c-param C_PARAM     Concentration parameter of the conditional
                        distribution.
  --m-param M_PARAM     Additional parameter for the marginal (only relevant
                        if it is not uniform).
  --tau TAU
  --n-mixing-layer N_MIXING_LAYER
                        Number of layers in nonlinear mixing network g.
  --n N                 Dimensionality of the latents.
  --space-type {box,sphere,unbounded}
  --m-p M_P             Type of ground-truth marginal distribution. p=0 means
                        uniform; all other p values correspond to (projected)
                        Lp Exponential
  --c-p C_P             Exponent of ground-truth Lp Exponential distribution.
  --lr LR
  --p P                 Exponent of the assumed model Lp Exponential
                        distribution.
  --batch-size BATCH_SIZE
  --n-log-steps N_LOG_STEPS
  --n-steps N_STEPS
  --resume-training
```

### KITTI Masks

```bash
>python main_kitti.py --help
usage: main_kitti.py [-h] [--box-norm BOX_NORM] [--p P] [--experiment-dir EXPERIMENT_DIR] [--evaluate] [--specify SPECIFY] [--random-search] [--random-seeds] [--seed SEED] [--beta BETA] [--gamma GAMMA]
                     [--rate-prior RATE_PRIOR] [--data-distribution DATA_DISTRIBUTION] [--rate-data RATE_DATA] [--data-k DATA_K] [--betavae] [--search-beta] [--output-dir OUTPUT_DIR] [--log-dir LOG_DIR]
                     [--ckpt-dir CKPT_DIR] [--max-iter MAX_ITER] [--dataset DATASET] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--image-size IMAGE_SIZE] [--use-writer] [--z-dim Z_DIM] [--lr LR]
                     [--beta1 BETA1] [--beta2 BETA2] [--ckpt-name CKPT_NAME] [--log-step LOG_STEP] [--save-step SAVE_STEP] [--kitti-max-delta-t KITTI_MAX_DELTA_T] [--natural-discrete] [--verbose] [--cuda]
                     [--num_runs NUM_RUNS]

Disentanglement with InfoNCE/Contrastive Learning - KITTI Masks

optional arguments:
  -h, --help            show this help message and exit
  --box-norm BOX_NORM
  --p P
  --experiment-dir EXPERIMENT_DIR
                        specify path
  --evaluate            evaluate instead of train
  --specify SPECIFY     use argument to only compute a subset of metrics
  --random-search       whether to random search for params
  --random-seeds        whether to go over random seeds with UDR params
  --seed SEED           random seed
  --beta BETA           weight for kl to normal
  --gamma GAMMA         weight for kl to laplace
  --rate-prior RATE_PRIOR
                        rate (or inverse scale) for prior laplace (larger -> sparser).
  --data-distribution DATA_DISTRIBUTION
                        (laplace, uniform)
  --rate-data RATE_DATA
                        rate (or inverse scale) for data laplace (larger -> sparser). (-1 = rand).
  --data-k DATA_K       k for data uniform (-1 = rand).
  --betavae             whether to do standard betavae training (gamma=0)
  --search-beta         whether to do rand search over beta
  --output-dir OUTPUT_DIR
                        output directory
  --log-dir LOG_DIR     log directory
  --ckpt-dir CKPT_DIR   checkpoint directory
  --max-iter MAX_ITER   maximum training iteration
  --dataset DATASET     dataset name (dsprites, cars3d,smallnorb, shapes3d, mpi3d, kittimasks, natural
  --batch-size BATCH_SIZE
                        batch size
  --num-workers NUM_WORKERS
                        dataloader num_workers
  --image-size IMAGE_SIZE
                        image size. now only (64,64) is supported
  --use-writer          whether to use a log writer
  --z-dim Z_DIM         dimension of the representation z
  --lr LR               learning rate
  --beta1 BETA1         Adam optimizer beta1
  --beta2 BETA2         Adam optimizer beta2
  --ckpt-name CKPT_NAME
                        load previous checkpoint. insert checkpoint filename
  --log-step LOG_STEP   numer of iterations after which data is logged
  --save-step SAVE_STEP
                        number of iterations after which a checkpoint is saved
  --kitti-max-delta-t KITTI_MAX_DELTA_T
                        max t difference between frames sampled from kitti data loader.
  --natural-discrete    discretize natural sprites
  --verbose             for evaluation
  --cuda
  --num_runs NUM_RUNS   when searching over seeds, do 10
```

### 3DIdent

```bash
>python main_3dident.py --help
usage: main_3dident.py [-h] [--batch-size BATCH_SIZE] [--n-eval-samples N_EVAL_SAMPLES] [--lr LR] [--optimizer {adam,sgd}] [--iterations ITERATIONS]
                                                                   [--n-log-steps N_LOG_STEPS] [--load-model LOAD_MODEL] [--save-model SAVE_MODEL] [--save-every SAVE_EVERY] [--no-cuda] [--position-only]
                                                                   [--rotation-and-color-only] [--rotation-only] [--color-only] [--no-spotlight-position] [--no-spotlight-color] [--no-spotlight]
                                                                   [--non-periodic-rotation-and-color] [--dummy-mixing] [--identity-solution] [--identity-mixing-and-solution]
                                                                   [--approximate-dataset-nn-search] --offline-dataset OFFLINE_DATASET [--faiss-omp-threads FAISS_OMP_THREADS]
                                                                   [--box-constraint {None,fix,learnable}] [--sphere-constraint {None,fix,learnable}] [--workers WORKERS]
                                                                   [--mode {supervised,unsupervised,test}] [--supervised-loss {mse,r2}] [--unsupervised-loss {l1,l2,l3,vmf}]
                                                                   [--non-periodical-conditional {l1,l2,l3}] [--sigma SIGMA] [--encoder {rn18,rn50,rn101,rn151}]

Disentanglement with InfoNCE/Contrastive Learning - 3DIdent

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --n-eval-samples N_EVAL_SAMPLES
  --lr LR
  --optimizer {adam,sgd}
  --iterations ITERATIONS
                        How long to train the model
  --n-log-steps N_LOG_STEPS
                        How often to calculate scores and print them
  --load-model LOAD_MODEL
                        Path from where to load the model
  --save-model SAVE_MODEL
                        Path where to save the model
  --save-every SAVE_EVERY
                        After how many steps to save the model (will always be saved at the end)
  --no-cuda
  --position-only
  --rotation-and-color-only
  --rotation-only
  --color-only
  --no-spotlight-position
  --no-spotlight-color
  --no-spotlight
  --non-periodic-rotation-and-color
  --dummy-mixing
  --identity-solution
  --identity-mixing-and-solution
  --approximate-dataset-nn-search
  --offline-dataset OFFLINE_DATASET
  --faiss-omp-threads FAISS_OMP_THREADS
  --box-constraint {None,fix,learnable}
  --sphere-constraint {None,fix,learnable}
  --workers WORKERS     Number of workers to use (0=#cpus)
  --mode {supervised,unsupervised,test}
  --supervised-loss {mse,r2}
  --unsupervised-loss {l1,l2,l3,vmf}
  --non-periodical-conditional {l1,l2,l3}
  --sigma SIGMA         Sigma of the conditional distribution (for vMF: 1/kappa)
  --encoder {rn18,rn50,rn101,rn151}
```

# 3DIdent Dataset
We introduce *3DIdent*, a dataset with hallmarks of natural environments (shadows, different lighting conditions, 3D rotations, etc.).
<p align="center">
  <img src="https://brendel-group.github.io/cl-ica/img/3ddis.svg" alt="3DIdent dataset example images" />
</p>

You can access the full dataset [here](https://zenodo.org/record/4502485). The training and test datasets consists of 250000 and 25000 samples, respectively. To load, you can use the `ThreeDIdentDataset` class defined in [datasets/threedident_dataset.py](datasets/threedident_dataset.py).

## BibTeX
If you find our analysis helpful, please cite our pre-print:

```bibtex
@article{zimmermann2021cl,
  author = {
    Zimmermann, Roland S. and
    Sharma, Yash and
    Schneider, Steffen and
    Bethge, Matthias and
    Brendel, Wieland
  },
  title = {
    Contrastive Learning Inverts
    the Data Generating Process
  },
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
    {ICML} 2021, 18-24 July 2021, Virtual Event},
  series = {Proceedings of Machine Learning Research},
  volume = {139},
  pages = {12979--12990},
  publisher = {{PMLR}},
  year = {2021},
  url = {http://proceedings.mlr.press/v139/zimmermann21a.html},
}
```
