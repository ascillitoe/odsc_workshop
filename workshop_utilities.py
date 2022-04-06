"""
DO NOT EDIT.

This file contains utility functions and classes for the ODSC workshop "An Introduction to Drift Detection". It is used by "intro_to_drift_detection.ipynb".

To fetch prerequisites for the workshop in advance, run this script with "python workshop_utilities.py". 
"""
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import umap
from typing import Union, Optional, List, Dict
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm
from pathlib import Path
from urllib.error import URLError
from alibi_detect.utils.pytorch.data import TorchDataset
from torch.utils.data import DataLoader
import seaborn as sns
import subprocess


# Not a recommended practice, but we will hard code device as a global variable here
# so that it is easy to change if required (this also needs to be set in the notebook). 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CACHE_FOLDER = Path('cache/')


def fetch_prerequisites():
    """
    Downloads the neccesary data and models for the workshop. They are saved 
    to disk so that the workshop notebook can be run offline at a later date.
    """
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True) 

    # Download the 20newsgroup dataset with scikit-learn
    try:
        dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, data_home=CACHE_FOLDER)
    except URLError as e:
        raise Exception('Download failed, check your internet connection') from e

    # Load the sentence transformer model
    # (see https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
    try:
        filepath = CACHE_FOLDER.joinpath('sentence-transformers_paraphrase-MiniLM-L6-v2')
        sentence_transformer = SentenceTransformer(filepath)
    except:
        try:
            sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=CACHE_FOLDER)
        except:
            print('Download failed, check your internet connection')

    return dataset, sentence_transformer


class EmbeddingModel:
    """
    A pytorch embedding model. This is a wrapper around a pre-trained sentence transformer model. 
    It transforms/encodes sentences and paragraphs into 384-dimensional dense vectors.
    """
    def __init__(
        self,
        model: Union[str, SentenceTransformer] = 'paraphrase-MiniLM-L6-v2',  # https://www.sbert.net/docs/pretrained_models.html
        max_seq_length: int = 200,
        batch_size: int = 32
    ) -> None:
        if isinstance(model, str):  # model still a str, pass to SentenceTransformer
            model = SentenceTransformer(model)
        self.text_encoder = model.to(device)
        self.text_encoder.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.text_encoder.encode(x, convert_to_tensor=True, batch_size=self.batch_size,
                                       show_progress_bar=False)


class Classifier(nn.Module):
    """
    A pytorch classifier model to classify sentences/paragraphs into `n_classes` number of topics.
    The backbone of the model is a pre-trained sentence transformer model, and simple MPL head is 
    added for classification. Only the MLP head is trained.
    """

    def __init__(
        self, 
        embedding: EmbeddingModel = EmbeddingModel(),
        max_seq_length: int = 200,
        n_classes: int = 20
    ) -> None:
        """ Text classification model. Note that we do not train the embedding backbone."""
        super().__init__()
        self.embedding_model = embedding
        self.embedding_model.text_encoder.max_seq_length = max_seq_length
        for param in self.embedding_model.text_encoder.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
                nn.Linear(384, 256), 
                nn.LeakyReLU(.1), 
                nn.Dropout(.5), 
                nn.Linear(256, n_classes)
                )
        
    def forward(self, x: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):  # If x is still a list of strings, encode with embedding_model
            x = self.embedding_model(x)
        elif isinstance(x, np.ndarray):  # self.head() expects Tensor
            x = torch.from_numpy(x).to(device)
        return self.head(x).detach()


class UMAPModel:
    """
    A UMAP model class. This performs dimension reduction, returning a projection 
    of the data onto a low dimensional embedding amenable to visualisation.

    UMAP =  Uniform Manifold Approximation and Projection for Dimension Reduction.
    https://umap-learn.readthedocs.io/en/latest/
    """
    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 2,
        metric: str = 'euclidean',
        min_dist: float = .1,
        **kwargs: dict
    ) -> None:
        super().__init__()
        kwargs = kwargs if isinstance(kwargs, dict) else dict()
        kwargs.update(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist
        )
        self.model = umap.UMAP(**kwargs)
        self.fitted = False

    def fit(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """ Fit UMAP embedding. A combination of labeled and unlabeled data
        can be passed. Unlabeled instances are equal to -1. """
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        self.model.fit(x, y=y)
        self.fitted = True

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """ Transform the input x to the embedding space. """
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        return self.model.transform(x)


def train_model(model, x, y, epochs=3, lr=1e-3, batch_size=32, shuffle=False):
    """
    Trains a classifier model by minimizing cross entropy loss with
    the Adam optimizer.
    """
    loader = DataLoader(TorchDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in tqdm(loader):
            y = y.to(device)
            y_hat = model(list(x))  # loader returns x as a tuple but model expects list
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()


def eval_model(model, x, y, verbose=1, batch_size=32, shuffle=False):
    """
    Evaluates a classifier's accuracy score, returns logits and predictions.
    """
    loader = DataLoader(TorchDataset(x, y), batch_size=batch_size, shuffle=shuffle)
    logits, labels = [], []
    with torch.no_grad():
        if verbose == 1:
            loader = tqdm(loader)
        for x, y in loader:
            y_hat = model(list(x))  # loader returns x as a tuple but model expects list
            logits += [y_hat.cpu().numpy()]
            labels += [y.cpu().numpy()]
    logits = np.concatenate(logits, 0)
    preds = np.argmax(logits, 1)
    labels = np.concatenate(labels, 0)
    if verbose == 1:
        accuracy = (preds == labels).mean()
        print(f'Accuracy: {accuracy:.3f}')
    return logits, preds


def plot_clusters(x: np.ndarray, y: np.ndarray, 
        classes: list, dr_model: Optional[UMAPModel] = None, title: str = None) -> None:
    """
    Visualises the clustered text data by plotting data over the UMAP embedding, 
    and colouring by class.
    """
    if dr_model is not None:
        if not dr_model.fitted:
            raise ValueError('The given UMAPModel has not been fitted.')
        else:
            x = dr_model.predict(x)
    if x.shape[1] != 2:
        raise ValueError('Dimension of data to plot != 2')
    fig, ax = plt.subplots(1, figsize=(14, 10))
    # Plot
    x = x.T
    sns.scatterplot(x=x[0], y=x[1], hue=y, palette="deep")
    # Legend
    idx = np.unique(y)
    classes = np.array(classes)[idx]
    handles, labels  =  ax.get_legend_handles_labels()
    ax.legend(handles, classes, loc='center left', bbox_to_anchor=(1, 0.5))

    # Title
    if title:
        plt.title(title)


def split_data(x, y, n_ref_c, n_test_c, seed=None, y2=None, return_idx=False):
    """
    Samples the data into disjoint reference and test sets, with a specified 
    number of instances per class (i.e. per news topic).
    """
    if seed:
        np.random.seed(seed)
    
    # split data by class
    n_c = len(np.unique(y))
    idx_c = {_: np.where(y == _)[0] for _ in range(n_c)}
    
    # convert nb instances per class to a list if needed
    n_ref_c = [n_ref_c] * n_c if isinstance(n_ref_c, int) else n_ref_c
    n_test_c = [n_test_c] * n_c if isinstance(n_test_c, int) else n_test_c
    
    # sample reference, test and held out data
    idx_ref, idx_test, idx_held = [], [], []
    for _ in range(n_c):
        idx = np.random.choice(idx_c[_], size=len(idx_c[_]), replace=False)
        idx_ref.append(idx[:n_ref_c[_]])
        idx_test.append(idx[n_ref_c[_]:n_ref_c[_] + n_test_c[_]])
        idx_held.append(idx[n_ref_c[_] + n_test_c[_]:])
    idx_ref = np.concatenate(idx_ref)
    idx_test = np.concatenate(idx_test)
    idx_held = np.concatenate(idx_held)
    x_ref, y_ref = [x[_] for _ in idx_ref], y[idx_ref]
    x_test, y_test = [x[_] for _ in idx_test], y[idx_test]
    x_held, y_held = [x[_] for _ in idx_held], y[idx_held]
    if y2 is not None:
        y_ref2, y_test2, y_held2 = y2[idx_ref], y2[idx_test], y2[idx_held]
        return (x_ref, y_ref, y_ref2), (x_test, y_test, y_test2), (x_held, y_held, y_held2)
    elif not return_idx:
        return (x_ref, y_ref), (x_test, y_test), (x_held, y_held)
    else:
        return idx_ref, idx_test, idx_held


def plot_qq(p_vals: np.ndarray, title: str) -> None:
    """
    Plot QQ-plots of p-value to evaluate detector calibration.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12,10))
    fig.suptitle(title)
    n = len(p_vals)
    for i in range(9):
        unifs = p_vals if i==4 else np.random.rand(n)
        sm.qqplot(unifs, uniform(), line='45', ax=axes[i//3,i%3])
        if i//3 < 2:
            axes[i//3,i%3].set_xlabel('')
        if i%3 != 0:
            axes[i//3,i%3].set_ylabel('')


def plot_hist(
    p_vals: List[np.ndarray],
    title: str,
    colors: List[str] = ['salmon', 'turquoise'],
    methods: List[str] = ['MMD', 'CA-MMD'],
    ylim: Optional[tuple] = None
):
    for p_val, method, color in zip(p_vals, methods, colors):
        sns.histplot(p_val, color=color, kde=True, label=f'{method}', binwidth=0.05, stat='probability')
        plt.legend(loc='upper right')
    plt.xlim(-0.02, 1.02)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel('Density')
    plt.xlabel('p-values')
    plt.title(title)
    plt.show()


def set_seed(seed: int) -> None:
    """
    Set the relevent random seeds.
    (this might not be enough to enforce determinism on GPU).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    fetch_prerequisites()
