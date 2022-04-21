"""Module to keep static function that should usually stay in the library module.

Examples: torch.stack, torch.argmax
"""
# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import torch

from .mpc_tensor import MPCTensor
from ...util import parallel_execution


def helper_argmax(
    x: MPCTensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = False,
    one_hot: bool = False,
) -> MPCTensor:
    """Compute argmax using pairwise comparisons. Makes the number of rounds fixed, here it is 2.

    This is inspired from CrypTen.

    Args:
        x (MPCTensor): the MPCTensor on which to compute helper_argmax on
        dim (Union[int, Tuple[int]): compute argmax over a specific dimension(s)
        keepdim (bool): when one_hot is true, keep all the dimensions of the tensor
        one_hot (bool): return the argmax as a one hot vector

    Returns:
        Given the args, it returns a one hot encoding (as an MPCTensor) or the index
        of the maximum value

    Raises:
        ValueError: In case more max values are found and we need to return the index
    """
    # for each share in MPCTensor
    #   do the algorithm portrayed in paper (helper_argmax_pairwise)
    #   results in creating two matrices and subtraction them

    # TODO: Replace session everywhere acording to what we have
    prep_x = x.flatten() if dim is None else x
    args = [
        [str(uuid), share_ptr_tensor, dim]
        for uuid, share_ptr_tensor in zip(
            session.rank_to_uuid.values(), prep_x.share_ptrs
        )
    ]
    shares = parallel_execution(helper_argmax_pairwise, session.parties)(args)

    res_shape = shares[0].shape.get()
    x_pairwise = MPCTensor(shares=shares, session=x.session, shape=res_shape)

    # with the MPCTensor tensor we check what entries are positive
    # then we check what columns of M matrix have m-1 non-zero entries after comparison
    # (by summing over cols)
    pairwise_comparisons = x_pairwise >= 0

    # re-compute row_length
    _dim = -1 if dim is None else dim
    row_length = x.shape[_dim] if x.shape[_dim] > 1 else 2

    result = pairwise_comparisons.sum(0)
    result = result >= (row_length - 1)
    res_shape = res_shape[1:]  # Remove the leading dimension because of sum(0)

    if not one_hot:
        if dim is None:
            check = result * torch.Tensor([i for i in range(np.prod(res_shape))])
        else:
            size = [1 for _ in range(len(res_shape))]
            size[dim] = res_shape[dim]
            check = result * torch.Tensor([i for i in range(res_shape[_dim])]).view(
                size
            )

        if dim is not None:
            argmax = check.sum(dim=dim, keepdim=keepdim)
        else:
            argmax = check.sum()
            if (argmax >= row_length).reconstruct():
                # In case we have 2 max values, rather then returning an invalid index
                # we raise an exception
                raise ValueError("There are multiple argmax values")

        result = argmax

    return result


def argmax(
    x: MPCTensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim=False,
) -> MPCTensor:
    """Compute argmax using pairwise comparisons. Makes the number of rounds fixed, here it is 2.

    This is inspired from CrypTen.

    Args:
        x (MPCTensor): the MPCTensor that argmax will be computed on
        dim (Union[int, Tuple[int]): compute argmax over a specific dimension(s)
        keepdim (bool): when one_hot is true and dim is set, keep all the dimensions of the tensor

    Returns:
        The index of the maximum value as an MPCTensor
    """
    return helper_argmax(x, dim=dim, keepdim=keepdim, one_hot=False)


def max_mpc(
    x: MPCTensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = False,
    one_hot: bool = False,
) -> Union[MPCTensor, Tuple[MPCTensor, MPCTensor]]:
    """Compute the maximum value for an MPCTensor.

    Args:
        x (MPCTensor): MPCTensor to be computed the maximum value on.
        dim (Optional[Union[int, Tuple[int]]]): The dimension over which to compute the maximum.
        keepdim (bool): when one_hot is true and dim is set, keep all the dimensions of the tensor
        one_hot (bool): to return the maximum indices as a one hot tensor

    Returns:
        A tuple representing (max MPCTensor, indices_max MPCTensor)
    """
    argmax_mpc = helper_argmax(x, dim=dim, keepdim=keepdim, one_hot=True)
    max_mpc = argmax_mpc * x
    if dim is None:
        res = max_mpc.sum()
    else:

        if not one_hot:
            shape = argmax_mpc.shape
            size = [1 for _ in range(len(shape))]
            size[dim] = shape[dim]
            argmax_mpc = argmax_mpc * torch.Tensor([i for i in range(shape[dim])]).view(
                size
            )
            argmax_mpc = argmax_mpc.sum(dim=dim, keepdim=keepdim)
        max_mpc = max_mpc.sum(dim=dim, keepdim=keepdim)
        res = max_mpc, argmax_mpc

    return res


def helper_argmax_pairwise(
    share: ShareTensor, dim: Optional[Union[int, Tuple[int]]] = None
) -> ShareTensor:
    """Helper function that would compute the difference between all the elements in a tensor.

    Credits goes to the CrypTen team.

    Args:
        share (ShareTensor): Share tensor
        dim (Optional[Union[int, Tuple[int]]]): dimension to compute over

    Returns:
        A ShareTensor that represents the difference between each "row" in the ShareTensor.
    """
    # TODO: Replace session
    session = get_session(session_uuid_str)

    dim = -1 if dim is None else dim
    row_length = share.shape[dim] if share.shape[dim] > 1 else 2

    # Copy each row (length - 1) times to compare to each other row
    a = share.expand(row_length - 1, *share.shape)

    # Generate cyclic permutations for each row
    shares = torch.stack(
        [share.tensor.roll(i + 1, dims=dim) for i in range(row_length - 1)]
    )
    b = ShareTensor()
    b.tensor = shares

    return a - b


STATIC_FUNCS: Dict[str, Callable] = {
    "argmax": argmax,
    "max": max_mpc,
}
