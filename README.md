# Multilingual Document Classification with mBERT

## Installation
Prerequisites: `pip` and `python 3.7`. Using `anaconda3` or `miniconda3` is highly recommended, and this documentations will assume that you use it.

To run this software on GPU (recommended), install PyTorch 1.4.0 or PyTorch 1.5.0, depending on your version of CUDA (see https://pytorch.org for detailed instructions). To run on CPU, install PyTorch using the following command:
```
conda install pytorch torchvision cpuonly -c pytorch
```

After installing PyTorch, install the other requirements by running `pip install -r requirements.txt`.


## Training
The training scripts are located in `scripts/tobacco`. Run these from the base directory and not from the `scripts` or `scripts/tobacco` directories. In particular, the multilingual GPU model is trained using `run-tobacco-one.sh`. The best settings have been pre-selected (batch size 32, learning rate 5e-5, 12 epochs), but you may also train the model with various settings by uncommenting the hyperparameter loops before (and after) the call to `run_classifier.py`. A CPU-compatible version has also been provided as `run-tobacco-one-cpu.sh`.

These will save a model to a new directory named `model`. In particular, each specific model will have its own directory where evaluation results on the development set and the model parameters are stored. If you end up training many models with different hyperparameters, run the `src/find_best.py` script from the base directory to find the best one.

On a GTX 1080Ti, the best multilingual model took just under 20 minutes to train. On a CPU, it took 48 hours to train.


## Evaluation
The evaluation script for the multilingual model is located at `scripts/tobacco/eval-tobacco-one.sh`. This script loads the multilingual model and evaluates its performance on each language individually, storing test scores in a series of text files located in the `eval/tobacco/tuneall-one` directory. Each directory in `tuneall-one` represents results for a different language.

A nearly identical script for CPU may be found at `scripts/tobacco/eval-tobacco-one-cpu.sh`; the only difference is that it does not define the environment variables needed for GPUs and uses the `--no_cuda` argument.
