"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import argparse

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '10.8.11.12'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--rank", type=int, default=0, help="batch size of the forward pass")

    args = parser.parse_args()

    init_process(rank=args.rank, size=2, fn=run)