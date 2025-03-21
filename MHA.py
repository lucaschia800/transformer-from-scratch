import torch
import torch.nn as nn
import torch.nn.functional as F


def MHA_wrapper(query, key, value, n_heads=1, causal=False):

    assert query.shape == key.shape == value.shape
    _, n_tok, n_embd = query.shape

    query = query.transpose(0,1)
    key = key.transpose(0,1)
    value = value.transpose(0,1)

    in_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device).repeat((3, 1))
    out_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device)

    attn_mask = None
    if causal:
        attn_mask = torch.tril(torch.ones(n_tok, n_tok, dtype=bool, device=key.device)).logical_not()

    out, _ = F.multi_head_attention_forward(
        query, key, value, n_embd, n_heads,
        in_proj_weight=in_proj_weight, in_proj_bias=None,
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
        out_proj_weight=out_proj_weight, out_proj_bias=None,
        attn_mask=attn_mask, need_weights=False,)

    return out.transpose(0,1)



def init_qkv_proj(n_embd:int):
    """
 
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    return (nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd))


def self_attention(Q, K, V, n_heads=1, causal=True):
    """
    Self-attention block.

    :return: A tensor containing the result of the self-attention operation.
    """
    assert Q.shape == K.shape == V.shape
    B, n_tok, n_embd = Q.size()

    if n_heads > 1:
         Q, K, V = split_heads_qkv(Q, K, V, n_heads)


    A = pairwise_similarities(Q, K)
    A = attn_scaled(A, n_embd, n_heads)



    if causal:
        mask = make_causal_mask(n_tok)
        A = apply_causal_mask(mask, A) 

    A = attn_softmax(A)
    y = compute_outputs(A, V)


    if n_heads > 1:
        y = merge_heads(y)

    # output should have the same shape as input
    assert y.shape == (B, n_tok, n_embd)
    return y


def pairwise_similarities(Q, K):
    """
    Dot product attention is computed via the dot product between each query and each key.
    :return: The raw attention scores, A = QK^T.
    """

    A = Q @ K.transpose(-2, -1)
    return A

def attn_scaled(A, n_embd:float, n_heads:float):
    """
    Scale the raw attention scores.
    :return: Scaled raw attention scores.
    """
    assert n_embd % n_heads == 0, "d must be divisible by number of heads"

    A_scaled = A / (n_embd / n_heads)**0.5

    return A_scaled

def attn_softmax(A):
    """
    Normalize the scaled raw attention scores with softmax.
    :return: Normalized attention scores, A' = softmax(A).
    """

    A_softmax = F.softmax(A, dim=-1)
    return A_softmax

def compute_outputs(A, V):
    """
    Get outputs as a weighted sum of values by attention scores, using matrices.
    :return: Output, O = AV.
    """

    O = A @ V
    return O

def make_causal_mask(n_tok:int):
    """
    Create a mask matrix that masks future context for the attention.
    :return: A mask matrix which is a tensor of shape (n_tok, n_tok)
    """
    # Hint: In order for it to run properly later, you'll need to put `.to(DEVICE)` at
    # the end of your expression for this. This will not be relevant until section 2.2.

    mask = torch.tril(torch.ones(n_tok, n_tok), diagonal = 0)
    return mask.bool()

def apply_causal_mask(mask, A):
    """
    Apply mask to attention.
    :return: A masked attention matrix.
    """

    inf_tensor = torch.full_like(A, float('-inf'))
    A_masked = torch.where(mask, A, inf_tensor)
    return A_masked


def split_heads_qkv(Q, K, V, n_heads:int):
    """
    Provided as a utility -- you can choose to not use it if you'd like.
    """
    return (split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads))

def split_heads(x, n_heads:int):
    """
    Splitting x across multiple heads.
    :return: A splitted x.
    """
    B, n_tok, n_embd = x.size()
    assert n_embd % n_heads == 0, "d must be divisible by number of heads"
    x = x.reshape(B , n_tok, n_heads, n_embd // n_heads).permute(0, 2, 1, 3)
    return x 

def merge_heads(y):
    """
    Reversing splitting action of y.
    :return: A merged y.
    """
    B, nh, n_tok, nc = y.size()

    y = y.permute(0, 2, 1, 3).reshape(B, n_tok, nh * nc)
    return y
    


if __name__ == "__main__":


    DEVICE = 'cpu'


    part1_n_tok = 10
    part1_n_emb = 6

    # generate fixed pseudo-random Q,K,V for testing attn function
    torch.manual_seed(447)

    # Initialize random testing Q,K,V
    part1_key = torch.randn(1, part1_n_tok, part1_n_emb)
    part1_value = torch.randn(1, part1_n_tok, part1_n_emb)
    part1_query = torch.randn(1, part1_n_tok, part1_n_emb)



    # Test the self-attention function
    out = self_attention(part1_query, part1_key, part1_value, n_heads=1, causal=False)
    print(out.shape)
    print(out)

    # Test the MHA_wrapper function
    out = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=1, causal=False)
    print(out.shape)
    print(out)
    print("All tests passed!")