import fire
from sentence_transformers import SentenceTransformer
from diskcache import Index
from flwr.server.strategy import FedAvg
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union

# local imports
from utils.dataset import load_datasets
from utils.FlowerClient import FlowerClient, set_parameters, get_parameters
from utils.cutil import cacheEvaluator


def initializeModel(taname):
    model = SentenceTransformer(taname)
    return model

all_rounds_models_cache = Index(".storage/cache/all_rounds_model")


class FLSimulation:
    def __init__(self, key, config, c_train_dls, c_val, c_test, server_data):
        self.config = config
        self.device = config["device"]
        self.client_epochs = config["client_epochs"]
        self.client_resources = {"num_cpus": 3, "num_gpus": 1}
        self.train_dls, self.client_vals, self.client_test = c_train_dls, c_val, c_test
        self.server_val_data, self.server_test_data = server_data
        self._setStrategy()
        self.round2results = {}
        self.round2client_avg_results = []
        self.global_model = None
        self.f1_score_global_model = None
        self.highest_f1_score = -1
        self.key = key

    def _getFit_Config(self, server_round: int):
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": self.client_epochs,  #
        }
        return config

    def _evaluateGlobalModel(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        loss = None

        net = initializeModel(self.config["tname"])
        net = net.cpu()
        set_parameters(net, parameters)

        # if server_round % 5 == 0:

        r = cacheEvaluator(
            net.to(self.device),
            self.server_val_data,
            self.server_test_data,
            self.config["batch_size"],
        )

        net = net.cpu()
        self.global_model = net

        if server_round == int(self.config["num_rounds"]) / 2:
            r[f"global_model_half-r{server_round}"] = net

        net = initializeModel(self.config["tname"])
        net = net.cpu()
        set_parameters(net, parameters)

        all_rounds_models_cache[self.key + f"-r{server_round}"] = (net, r)

        if r["test"]["F1"] > self.highest_f1_score:
            print(f'New Highest F1 Score {r["test"]["F1"]}')
            self.highest_f1_score = r["test"]["F1"]
            
            self.f1_score_global_model = net
            r["Best F1"] = self.highest_f1_score

        self.round2results[server_round] = r  # storing results to use later
        return loss, r

    def _getClient(self, cid) -> FlowerClient:
        net = initializeModel(self.config["tname"])
        return FlowerClient(
            cid,
            net,
            self.train_dls[int(cid)],
            self.client_vals[int(cid)],
            self.client_test[int(cid)],
            self.device,
            self.config["batch_size"],
            self.config["loss_type"],
        )

    def _setStrategy(self):
        self.strategy = FedAvg(
            fraction_fit=0.0,
            min_fit_clients=self.config["clients_per_round"],
            fraction_evaluate=0.0,  # Sample 5% of available clients for evaluation
            min_evaluate_clients=0,
            on_fit_config_fn=self._getFit_Config,
            # fit_metrics_aggregation_fn=self.cleintFitAggMetrics,  # aggregates federated metrics
            # evaluate_metrics_aggregation_fn = test
            evaluate_fn=self._evaluateGlobalModel,
        )

    # def cleintFitAggMetrics(self, c2metrics):
    #     # print(f"len of metrics: {len(c2metrics)}")
    #     # print(f'len of nk: {len(nk)}')
    #     # print("metrics: ", c2metrics)
    #     # print("nk: ", nk)

    #     val = c2metrics[0][1]['val']
    #     test = c2metrics[0][1]['test']

    #     avg_val = {k: 0 for k in val.keys()}
    #     avg_test = {k: 0 for k in test.keys()}
    #     for i in range(len(c2metrics)):
    #         for k in val.keys():
    #             avg_val[k] += c2metrics[i][1]['val'][k]
    #         for k in test.keys():
    #             avg_test[k] += c2metrics[i][1]['test'][k]

    #     to2printval = {'F1': round(avg_val['f1'] / len(c2metrics), 4),  'Precision': round(avg_val['precision'] / len(c2metrics), 4), 'Recall': round(avg_val['recall'] / len(c2metrics), 4), 'Threshold': round(avg_val['f1_threshold'] / len(c2metrics), 4)}

    #     to2printtest = {'F1': round(avg_test['F1'] / len(c2metrics), 4),  'Precision': round(avg_test['Precision'] / len(c2metrics), 4), 'Recall': round(avg_test['Recall'] / len(c2metrics), 4), 'Threshold': round(avg_test['Threshold'] / len(c2metrics), 4)}

    #     print(f"avg_cval: {to2printval}")
    #     print(f"avg_ctest: {to2printtest}")

    #     final_dict = {'avg_cval': to2printval, 'avg_ctest': to2printtest}

    #     self.round2client_avg_results.append((final_dict, c2metrics))

    #     # _ = input("Press any key to continue")

    #     return final_dict

    def run(self):
        fl.simulation.start_simulation(
            client_fn=self._getClient,
            client_resources=self.client_resources,
            num_clients=self.config["num_clients"],
            config=fl.server.ServerConfig(num_rounds=self.config["num_rounds"]),
            strategy=self.strategy,
            ray_init_args={"num_cpus": 32},
        )


def main(
    tname,
    dname,
    num_clients,
    clients_per_round,
    batch_size,
    device,
    client_epochs,
    num_rounds,
    loss_type,
):
    rounds_cache = Index(".storage/cache/rounds")
    dataset_cache = Index(".storage/cache/datasets")

    global_model_cache = Index(".storage/cache/global_models")
    # temp_config = {"tname": 'paraphrase-MiniLM-L3-v2', "dname": "quora",
    #                "num_clients": 100, "num_rounds": 3, "batch_size": 32,
    #                'device': "cuda", "client_epochs": 1} # for testing

    config = {
        "tname": tname,
        "dname": dname,
        "clients_per_round": clients_per_round,
        "num_clients": num_clients,
        "batch_size": batch_size,
        "device": device,
        "client_epochs": client_epochs,
        "num_rounds": num_rounds,
        "loss_type": loss_type,
    }

    train_dls, cvals_data, ctest_data, server_data = None, None, None, None
    dataset_clients_key = (
        f"{config['dname']}-{config['num_clients']}-{config['batch_size']}"
    )
    if dataset_clients_key not in dataset_cache:
        train_dls, cvals_data, ctest_data, server_data = load_datasets(
            dname=config["dname"],
            num_clients=config["num_clients"],
            train_batch_size=config["batch_size"],
        )
        dataset_cache[dataset_clients_key] = (train_dls, cvals_data, ctest_data)
        print(f"> Miss: Dataset Distribution saved to cache: {dataset_clients_key}")
    else:
        train_dls, cvals_data, ctest_data = dataset_cache[dataset_clients_key]
        _, _, _, server_data = load_datasets(
            dname=config["dname"],
            num_clients=config["num_clients"],
            train_batch_size=config["batch_size"],
        )  # server does not need to be distributed and stored in cache
        print(f"> Hit: Dataset Distribution found in cache: {dataset_clients_key}")

    exp_key = "last:"
    for k, v in config.items():
        exp_key += f"{k}-{v}-"

    print(f"Experiment Key: {exp_key}")

    if exp_key in rounds_cache:
        # print(cache[exp_key])
        print(f"> Key already exist simulation completed.")
        exit()

    print("> Starting Simulation ..")
    sim = FLSimulation(exp_key, config, train_dls, cvals_data, ctest_data, server_data)
    sim.run()
    rounds_cache[exp_key] = [(config, sim.round2results, sim.round2client_avg_results)]
    print(sim.round2results)
    global_model_cache[exp_key] = [
        sim.global_model.cpu(),
        sim.f1_score_global_model.cpu(),
    ]

    print("> Simulation Completed")


if __name__ == "__main__":
    print("Starting Simulation ..")
    fire.Fire(main)
