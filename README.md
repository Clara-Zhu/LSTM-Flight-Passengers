# LSTM-Flight-Passengers
This repo provides a framework to develop, train, and evaluate LSTM Models using PyTorch.

## Environment
The code was developed as a PyCharm project, using the anaconda environment saved in `env.yml`.

## About the Project
* The models developed utilize a sequence of LSTM hidden cells, fed into a linear layer as is standard for outputting data.
* During training, a graph of the model and sample prediction is plotted into `Visuals`.
* `ModelEval.py` is intended to provide a more thorough evaluation of each saved model. current evaluation is uses time series cross-validation.<sup>1</sup>


## References
Reference | Citation | Notes
---- | ------- | ---
1 | Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. [OTexts.com/fpp2](OTexts.com/fpp2). Accessed on 6. Novemeber, 2020. | See section 3.4: Evaluating Forecast Accuracy
misc | Olah, Christopher. _Understanding LSTM Networks_. 27 August 2015. Online. 15 September 2020. [Link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | A strong introduction to foundation and concepts of Long Short-Term Memory
misc |  Neil, Wesley. _LSTMs in PyTorch: Understanding the LSTM Architecture and Data Flow_. 30 July 2020. Towards Data Science. Online. 1 November 2020.  [Link](https://towardsdatascience.com/lstms-in-pytorch-528b0440244). | Practical guide to apply LSTMs in PyTorch