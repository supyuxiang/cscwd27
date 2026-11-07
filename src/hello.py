import torch


a = torch.tensor(
    [
        [1,2,3,-1],
        [0.1,0.7,9,0],
    ]
)

topk_values, topk_indices = torch.topk(
    input=a,k=3,dim=-1
)

breakpoint()











