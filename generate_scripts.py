import os
import shlex
import fire


def trainScripts():
    # , "quora"
    dnames = ['quora']
    # tnames = ['multi-qa-mpnet-base-cos-v1', 'paraphrase-MiniLM-L3-v2', 'multi-qa-MiniLM-L6-cos-v1', 'multi-qa-distilbert-cos-v1']
    tnames_gptdataset = ['multi-qa-mpnet-base-cos-v1', "paraphrase-albert-small-v2"]
    num_clients = [25, 50, 75, 100, 125]
    num_rounds = 50
    batch_sizes = [128]
    device = "cuda"
    epochs = [6]
    clients_per_round_list = [4]
    loss_types = ["both-mnr-contrastive"]

    with open(f"_run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        for dname in dnames:
            for tname in tnames_gptdataset:
                for nc in num_clients:
                    for batch_size in batch_sizes:
                        if "mpnet" in tname:
                            batch_size = 128
                        for client_epochs in epochs:
                            for clients_per_round in clients_per_round_list:
                                for loss_type in loss_types:
                                    cmd = f"python fl_sim_train.py --tname {tname} --dname {dname} --clients_per_round {clients_per_round} --num_clients {nc} --batch_size {batch_size} --device {device} --client_epochs {client_epochs} --num_rounds {num_rounds} --loss_type {loss_type}"
                                    f.write(cmd + "\n")
                            # f.write("sleep 10\n")


if __name__ == "__main__":
    trainScripts()
