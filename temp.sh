# interact -A semcache -t 2:00:00 -p a100_normal_q --gres=gpu:1 -N1 --cpus-per-task=4 --ntasks-per-node=4

# dgx_dev_q
# a100_normal_q
# a100_dev_q

interact -A semcache -t 8:00:00 --ntasks-per-node=8
module load Anaconda3
conda activate llm


interact -A semcache -t 1:00:00 -p normal_q --gres=gpu:0 -N1 --cpus-per-task=16
 
test 1