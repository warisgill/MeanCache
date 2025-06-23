#!/bin/bash
python fl_sim_train.py --tname multi-qa-mpnet-base-cos-v1 --dname quora --clients_per_round 4 --num_clients 25 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname multi-qa-mpnet-base-cos-v1 --dname quora --clients_per_round 4 --num_clients 50 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname multi-qa-mpnet-base-cos-v1 --dname quora --clients_per_round 4 --num_clients 75 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname multi-qa-mpnet-base-cos-v1 --dname quora --clients_per_round 4 --num_clients 100 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname multi-qa-mpnet-base-cos-v1 --dname quora --clients_per_round 4 --num_clients 125 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname paraphrase-albert-small-v2 --dname quora --clients_per_round 4 --num_clients 25 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname paraphrase-albert-small-v2 --dname quora --clients_per_round 4 --num_clients 50 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname paraphrase-albert-small-v2 --dname quora --clients_per_round 4 --num_clients 75 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname paraphrase-albert-small-v2 --dname quora --clients_per_round 4 --num_clients 100 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
python fl_sim_train.py --tname paraphrase-albert-small-v2 --dname quora --clients_per_round 4 --num_clients 125 --batch_size 128 --device cuda --client_epochs 6 --num_rounds 50 --loss_type both-mnr-contrastive
