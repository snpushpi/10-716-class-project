# Description of datasets and analysis code

This repository contains the relevant codes and datasets for the 10-716 project final report. To be able to reproduce the results shared in the report, pls follow the instructions below - 

- `FNSPID/stock_news_author_int_final`: This folder contains the integrated data combining the news article Latent Sematic Analysis summaries, the scaled sentiment scores as provided from the authors of https://github.com/Zdong104/FNSPID_Financial_News_Dataset/tree/main, other numerical stock features along with the inferred Potency score metric on heldout data. The preprocessing files that were used to help us produce these final datasets are the python files residing in the FNSPID/ folder itself.

- `FNSPID/embedding_collection/`: This folder contains the relevant code to run the pipeline of generating embeddings, fitting a model and then inferring Potency score on the remaining datasets. To be able to collect the embeddings and fit model **M_1**, simply run the `embedding_model_fitting.py`. In order to use the model to predict Potency Score on unseen datasets, simply run `inference.py`. The shell files can be used to help with that. The fitted model is saved as `ridge_model.joblib`.

- `FNSPID/FNSPID_Financial_News_Dataset/dataset_test/`: This folder contains the main codes for fitting the transformer and timesnet models. They have been adopted from https://github.com/Zdong104/FNSPID_Financial_News_Dataset/tree/main.  `FNSPID/FNSPID_Financial_News_Dataset/dataset_test/TimesNet-for-Time-Series-Prediction` contains the relevant code for Timesnet model and `FNSPID/FNSPID_Financial_News_Dataset/dataset_test/Transformer-for-Time-Series-Prediction` contains the relevant code for Transformer model. Simply run the file named `run_experiments_for_report.py` in either of the folders to get the performances reported in the paper.

If you find this work useful, consider citing the original work!

```
@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series}, 
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}
```


