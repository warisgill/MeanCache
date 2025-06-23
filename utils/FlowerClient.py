from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
from sentence_transformers import losses, util
from sentence_transformers import SentenceTransformer, evaluation
import logging
import pandas as pd
from .cutil import testDataEvaluation


def _trainTransformer(model, num_epochs, all_datasets, batch_size, loss_type):
    assert loss_type in ["mnr-loss", "contrastive", "both-mnr-contrastive"]

    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    margin = 0.5

    train_dls, test_val, test_raw = all_datasets
    train_dl_mnr, train_dl_c = train_dls

    dev_sentences1, dev_sentences2, dev_labels = test_val
    test_sentences1, test_sentences2, test_labels = test_raw
    # df = pd.DataFrame({"question1": dev_sentences1, "question2": dev_sentences2, "is_duplicate": dev_labels})

    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
        dev_sentences1 + test_sentences1,
        dev_sentences2 + test_sentences2,
        dev_labels + test_labels,
        batch_size=batch_size,
    )

    train_loss_constrative = losses.OnlineContrastiveLoss(
        model=model, distance_metric=distance_metric, margin=margin
    )
    train_loss_multiplenegatives = losses.MultipleNegativesRankingLoss(model)

    epoch2score = []

    # def extractScore(score, epochs, steps):
    #     epoch2score.append(score)

    train_objectives = []

    if loss_type == "mnr-loss":
        train_objectives.append((train_dl_mnr, train_loss_multiplenegatives))
    elif loss_type == "contrastive":
        train_objectives.append((train_dl_c, train_loss_constrative))
    elif loss_type == "both-mnr-contrastive":
        train_objectives.append((train_dl_mnr, train_loss_multiplenegatives))
        train_objectives.append((train_dl_c, train_loss_constrative))

    model.train()
    model.fit(
        train_objectives=train_objectives,
        evaluator=binary_acc_evaluator,
        epochs=num_epochs,
        warmup_steps=1000,
        # output_path=model_save_path,
        # callback=extractScore,
        show_progress_bar=False,
    )

    # model.eval()
    # fine_tune_result = binary_acc_evaluator.compute_metrices(model)
    # fine_tune_result["cossim"]  = fine_tune_result["cossim"]
    # threshold = fine_tune_result["cossim"]["f1_threshold"]
    # test_sentences1, test_sentences2, test_labels = test_raw
    # test_result = testDataEvaluation(model, test_sentences1, test_sentences2, test_labels, threshold, batch_size, cache_hit_miss=True)

    # test evaluator
    model = model.cpu()
    return model, {}, {}

    # threshold = fine_tune_result["cossim"]["f1_threshold"]
    # # test_result  = testScoreCosineSimilarity(model, test_sentences1, test_sentences2, test_labels, threshold)
    # test_result["threshold"] = threshold
    # exp2detail["Test-Data-Result-Fine-Tune"] = test_result
    # exp2detail["epoch2score"] = epoch2score

    # logging.info("Evaluate model after fine-tuning ")
    # logging.info(f"Result-without-fine-tuning {fine_tune_result}")
    # exp2detail["Val-Fine-Tune-Result"] = fine_tune_result


def get_parameters(model):
    model.train()
    model = model.cpu()
    # Extract parameters as numpy arrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    model.train()
    model = model.cpu()
    params_dict = zip(model.state_dict().keys(), parameters)
    # Create a new state_dict with the parameters, ensuring tensors are on the correct device
    state_dict = OrderedDict()

    for k, v in params_dict:
        # Ensure the tensor is on the same device as the model and preserve the data type
        dtype = model.state_dict()[k].dtype
        state_dict[k] = torch.tensor(v, dtype=dtype, device=model.device)

    # Load the new state_dict into the model
    model.load_state_dict(state_dict, strict=False)  # Set strict to False if necessary


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, raw_val, raw_test, device, batch_size, loss_type):
        self.cid = cid
        self.batch_size = batch_size
        self.net = net
        self.train_dl_mlr, self.train_dl_cl = trainloader
        self.raw_val = raw_val
        self.raw_test = raw_test
        self.DEVICE = device
        self.loss_type = loss_type

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        set_parameters(self.net, parameters)
        self.net = self.net.to(self.DEVICE)
        self.net, val_results, test_results = _trainTransformer(
            self.net,
            local_epochs,
            ((self.train_dl_mlr, self.train_dl_cl), self.raw_val, self.raw_test),
            self.batch_size, self.loss_type
        )

        self.net = self.net.cpu()

        nk_client_data_points = len(self.train_dl_cl)

        parameters = get_parameters(self.net)

        result = {
            "val": val_results,
            "test": test_results,
        }

        return parameters, nk_client_data_points, result

    def evaluate(self, parameters, config):
        return -1.1, -1, {}
