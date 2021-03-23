import argparse

def vqvae_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--num_hiddens', type=int, default=128)
    parser.add_argument('--n_res_hiddens', type=int, default=64)
    parser.add_argument('--n_res_block', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--n_embedding', type=int, default=256)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--sched', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    args = parser.parse_args()
    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args
