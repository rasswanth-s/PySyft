from typing import List, Tuple, Any
import unittest
from uuid import uuid1
import numpy as np
import copy

import torch 
from torch.testing._internal.common_utils import TestCase, run_tests

numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float32,  # float64 -> float32
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, DecorateInfo
from torch.utils._pytree import tree_map

import syft as sy
from syft.core.tensor.autodp.phi_tensor import TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer
sy.logger.remove()

aten = torch.ops.aten

np_tensor_backend_impl = dict()

def numpy_to_tensor_pointer(np_arr):
    owner, user = syf_login()
    backend_phi_tensor = sy.Tensor(np_arr)
    assert np_arr.ndim != 0
    data_subjects = ["abc"] * np_arr.shape[0] if np_arr.ndim != 0 else ["abc"]
    single_data_subject = backend_phi_tensor.private(
        min_val=10,
        max_val=90,
        data_subjects=data_subjects)
    data_asset_name = str(uuid1())
    # owner.load_dataset(
    #     assets={
    #         data_asset_name: single_data_subject,
    #     },
    #     name="my_data",
    #     description="description"
    # )
    # tensor_pointer = user.datasets[-1][data_asset_name]
    tensor_pointer = single_data_subject.send(owner)
    assert isinstance(tensor_pointer, TensorWrappedPhiTensorPointer)
    return tensor_pointer

def register_func(name, ns):
    # Helpful decorator to colocate registration and definition
    def register_func_inner(f):
        assert name not in ns, f"{name} has already been registered"
        ns[name] = f
        return f
    return register_func_inner

def torch_to_numpy_dtype_dict_int(dtype_int):
    # FIXME: Looks like this has been fixed by core, I don't think we need this anymore
    if dtype_int == torch.float32:
        return np.float64
    else:
        assert False, f"got: {dtype_int=}"

def numpy_to_torch(t):
    # Replacement for torch.from_numpy that actually handles scalars
    if np.isscalar(t):
        return torch.tensor(t)
    else:
        return torch.from_numpy(t)

@register_func(aten.add.Tensor, np_tensor_backend_impl)
def np_tensor_add_impl(x, y):  # @fix_binary_type_promotion
    return x + y

@register_func(aten.sub.Tensor, np_tensor_backend_impl)
def np_tensor_sub_impl(x, y):  # @fix_binary_type_promotion
    return x - y

@register_func(aten.rsub.Scalar, np_tensor_backend_impl)
def np_tensor_rsub_impl(x, y):  # @fix_binary_type_promotion
    return y - x

@register_func(aten.mul.Tensor, np_tensor_backend_impl)
def np_tensor_mul_impl(x, y):  # @fix_binary_type_promotion
    return x * y

@register_func(aten.div.Tensor, np_tensor_backend_impl)
def np_tensor_div_impl(x, y):  # @fix_binary_type_promotion
    return x / y

@register_func(aten.sum.default, np_tensor_backend_impl)
def np_tensor_sum_default_impl(x, keepdims=False):
    assert not keepdims
    return x.sum()

@register_func(aten.sum.dim_IntList, np_tensor_backend_impl)
def np_tensor_sum_dim_IntList_impl(x, dims, keepdims=False):
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    if len(dims) == 1 and x.public_shape[dims[0]] == 1 and keepdims:
        return x + 0
    np_tensor = x.publish(sigma=1e-4).block_with_timeout(10).get()
    return numpy_to_tensor_pointer(np.sum(np_tensor, axis=dims, keepdims=keepdims))

@register_func(aten.gt.Scalar, np_tensor_backend_impl)
def np_tensor_gt_impl(x, y):
    return x > y

@register_func(aten.reciprocal.default, np_tensor_backend_impl)
def np_tensor_reciprocal_impl(x):
    return x.reciprocal()

@register_func(aten.exp.default, np_tensor_backend_impl)
def np_tensor_exp_impl(x):
    return x.exp()

@register_func(aten.mm.default, np_tensor_backend_impl)
def np_tensor_mm_impl(x, y):
    ret = x @ y
    return ret

# View Ops

@register_func(aten.view.default, np_tensor_backend_impl)
def np_tensor_view_impl(x, shape):
    if x.public_shape == shape:
        return x
    np_tensor = x.publish(sigma=1e-4).block_with_timeout(10).get()
    ret = np_tensor.view()
    ret.shape = shape
    return numpy_to_tensor_pointer(ret)

@register_func(aten.expand.default, np_tensor_backend_impl)
def np_tensor_expand_impl(x, out_size):
    # FIXME: we don't have views yet, so just make a new tensor for now
    np_tensor = x.publish(sigma=1e-4).block_with_timeout(10).get()
    return numpy_to_tensor_pointer(np.broadcast_to(np_tensor, out_size))

@register_func(aten.t.default, np_tensor_backend_impl)
def np_tensor_t_impl(x):
    # Question: This is a view?
    return x.T

@register_func(aten.detach.default, np_tensor_backend_impl)
def np_tensor_detach_impl(x):
    # To the backend, detach is simply a view (FIXME: but we don't have views yet
    # so, just do a "clone" because the shape is the same)
    return x + 0

# Casting ops

@register_func(aten._to_copy.default, np_tensor_backend_impl)
def np_tensor__to_copy_impl(x, dtype):
    return x + 0  # FIXME: cast -> clone

# Factory ops

@register_func(aten.ones_like.default, np_tensor_backend_impl)
def np_tensor_ones_like_impl(x, dtype, layout, device, pin_memory, memory_format):
    
    return x.ones_like()

decompositions = dict()

@register_func(aten.relu.default, decompositions)
def relu(x):
    return x * (x > 0)

@register_func(aten.addmm.default, decompositions)
def addmm(bias, a, b):
    return torch.mm(a, b) + bias

@register_func(aten.sigmoid.default, decompositions)
def sigmoid(x):
    return 1 / (1 + torch.exp(x * -1))

@register_func(aten.sigmoid_backward.default, decompositions)
def sigmoid_backward(grad_out, result):
    # FIXME: DPTensor needs to be on the LHS always?
    return grad_out * result * (result * -1 + 1)

@register_func(aten.threshold_backward.default, decompositions)
def threshold_backward(grad_out, self, threshold):
    return grad_out * (self > threshold)

def contig_strides_from_shape(shape):
    return torch.zeros(shape).stride()

class DPTensor(torch.Tensor):
    # This class wraps a TensorPointer so autograd, and other can be used
    #
    # TODO:
    # - Think about the gamma case
    # - nit: maybe use slots here to be more efficient
    @staticmethod
    def __new__(cls, pointer, requires_grad=False):
        assert isinstance(pointer, TensorWrappedPhiTensorPointer) or \
            isinstance(pointer, TensorWrappedGammaTensorPointer)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            pointer.public_shape,
            strides=contig_strides_from_shape(pointer.public_shape),
            storage_offset=0,  # elem.storage_offset(),
            # TODO: clone storage aliasing
            # FIXME: why don't we just return a np type instead of a string?
            dtype=numpy_to_torch_dtype_dict[getattr(np, str(pointer.public_dtype))],
            layout=torch.strided,
            device="cpu",  # elem.device
            requires_grad=requires_grad
        )
        r.pointer = pointer

        return r
        
    __torch_function__ = torch._C._disabled_torch_function_impl
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f"__torch_dispatch__: {func.__module__}.{func.__name__}")
        def wrap(t):
            if isinstance(t, TensorWrappedPhiTensorPointer):
                return DPTensor(t)
            elif isinstance(t, TensorWrappedGammaTensorPointer):
                return DPTensor(t)
            else:
                assert False, f"__torch_dispatch__ wrap got unexpected type: {type(t)}" 
                # return t
        def unwrap(t):
            if isinstance(t, DPTensor):
                return t.pointer
            elif isinstance(t, torch.Tensor):
                # Sometimes we get tensors (either because autograd creates tensors internally)
                # or scalars get wrapped into tensors - wrapped number
                return t.detach().numpy()
            else:  # int, torch.dtype, etc? (we should have an explicit white-list)
                return t
            # else:
            #     assert False, f"Unsupported type: {type(t)}"
        if func in np_tensor_backend_impl:
            rets = tree_map(wrap, np_tensor_backend_impl[func](*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
            # get_logging_tensor_handler().emit(func.__name__, args, kwargs, rets)
            # print('\n'.join(get_logging_tensor_handler().log_list))
            return rets
        elif func in decompositions:
            return decompositions[func](*args, **kwargs)
        else:
            raise NotImplementedError(f"Backend has not implemented {func.__module__}.{func.__name__}")

    def __repr__(self):
        # TODO: maybe we should include autograd information?
        return repr(self.pointer)

    def synthetic(self):
        return self.pointer.synthetic
    
    def publish(self, sigma=None):
        ret = torch.from_numpy(self.pointer.publish(sigma=sigma).block_with_timeout(10).get()).to(self.dtype).reshape(self.shape)
        return ret
    
# 4) Testing

def get_op_list(tested_op_list: List[Tuple]):
    return [opinfo for opinfo in op_db if (opinfo.name, opinfo.variant_test_name) in tested_op_list]

# Some testing utils from funtorch repo (test/common_utils.py)

def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure,
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip("Skipped!"),
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped

def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)

_syf_owner = None
_syf_user = None

def syf_login():
    global _syf_owner
    global _syf_user

    if _syf_owner is not None:
        return _syf_owner, _syf_user

    _syf_owner = sy.login(email="info@openmined.org", password="changethis", port=8081)
    _syf_owner.datasets.purge(skip_check=True)
    email = str(uuid1())
    password = "pw"
    if email not in [entry['email'] for entry in _syf_owner.users.all()]:
        _syf_owner.users.create(
            **{
                "name": "Sheldon2 Cooper",
                "email": email,
                "password": password,
                "budget": 1e9
            }
        )
    _syf_user = sy.login(email=email, password=password, port=8081)
    return _syf_owner, _syf_user

def _allclose_with_type_promotion(a, b, rtol, atol):
    # FIXME: float32 is not supported by Syft
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)

class NPTensorTest(TestCase):
    def setUp(self):
        owner, user = syf_login()
        assert user.privacy_budget > 1e7

    def nptensorwrapper_to_torch(self, t, sigma=1e-4):
        # There will be noise!
        owner, user = syf_login()
        before = owner.privacy_budget
        ret = t.publish(sigma=sigma)
        after = owner.privacy_budget
        assert before - after > 1
        return ret

    def wrap_samples_with_subclass(self, samples):
        # Given OpInfo sample, reproduce sample, but with arguments wrapped
        # Have all the sample inputs in a single data set, where each
        # asset corresponds to a single sample
        # Returns tuple of just
        def handle_arg(t):
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.float32:
                    # FIXME: float32 is not supported by Syft
                    t = t.to(torch.float64)
                np_arr = t.detach().numpy()
                assert np_arr.ndim != 0
                backend_phi_tensor = sy.Tensor(np_arr)
                data_subjects = ["abc"] * np_arr.shape[0] if np_arr.ndim != 0 else ["abc"]
                single_data_subject = backend_phi_tensor.private(
                    min_val=10,
                    max_val=90,
                    data_subjects=data_subjects)
                return DPTensor(single_data_subject.send(owner))
            else:
                return t
        owner, _ = syf_login()
        all_returns = []
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs= sample.kwargs
            all_returns.append(([handle_arg(arg) for arg in args], {k: handle_arg(v) for k,v in kwargs.items()}))
        return all_returns


    def torch_to_nptensorwrapper(self, t, single_data_subject=True):
        if not single_data_subject:
            raise NotImplementedError

        if isinstance(t, torch.Tensor):
            if t.dtype == torch.float32:
                # FIXME: float32 is not supported by Syft
                t = t.to(torch.float64)
            np_arr = t.detach().numpy()
            tensor_pointer = numpy_to_tensor_pointer(np_arr)
            return DPTensor(tensor_pointer).requires_grad_(t.requires_grad)
        else:
            return t

   # 4.1) XOR Training Example

    def test_train_XOR(self):
        import torch.nn as nn
        import torch.nn.functional as F

        SIGMA = 0.4
        N_PER_QUAD = 100
        N_ITER = 2

        train_X_centers = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        train_y = [0, 1, 1, 0]

        shuffle_idx = np.arange(N_PER_QUAD * 4)
        np.random.shuffle(shuffle_idx)
        train_X_list = []
        train_y_list = []

        for center, label in zip(train_X_centers, train_y):
            train_X_list.append(np.random.randn(N_PER_QUAD, 2) * SIGMA + np.array(center))
            train_y_list.append(np.zeros(N_PER_QUAD) if label == 0 else np.ones(N_PER_QUAD))

        # Linear layer does not support double
        train_X_np = np.concatenate(train_X_list)[shuffle_idx].astype(np.float64)
        train_y_np = np.concatenate(train_y_list)[shuffle_idx].astype(np.float64)

        # Convert input data to torch.Tensor
        train_X = DPTensor(numpy_to_tensor_pointer(train_X_np))
        train_y = DPTensor(numpy_to_tensor_pointer(train_y_np))

        # Simple MLP

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = nn.Linear(2, 10)
                self.linear2 = nn.Linear(10, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.relu(x)
                return x.sigmoid()

        # Train

        model =  Net()
        sgd = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Register hook that publishes gradients (unwraps)
        for p in model.parameters():
            p.register_hook(lambda x: x.publish(sigma=1e-4))

        # For testing only
        model_copy = copy.deepcopy(model)
        train_X_tensor = torch.from_numpy(train_X_np).to(torch.float32)
        train_y_tensor = torch.from_numpy(train_y_np).to(torch.float32)
        old_params = [p.clone().detach() for p in model.parameters()]
        losses = []

        # Training loop
        for i in range(N_ITER):
            print("ITER: ", i)
            out = model(train_X)
            loss = (out - train_y).sum()
            loss.backward()

            # Testing computed gradients are the same
            out1 = model_copy(train_X_tensor)
            loss1 = (out1 - train_y_tensor).sum()
            loss1.backward()
            if i == 0:
                for p1, p2 in zip(model.parameters(), model_copy.parameters()):
                    assert torch.allclose(p1.grad, p2.grad, atol=3e-2, rtol=3e-2)

            sgd.step()
            sgd.zero_grad()
            losses.append(loss.detach())

        # Test parameters have been updated
        for p1, p2 in zip(old_params, model.parameters()):
            assert not torch.allclose(p1, p2)

        # Run test set

        test_X = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        test_y = torch.tensor([[0., 1., 1., 0.]])
        predicted = model(test_X)

        # Check whether we predictions are good
        # FIXME: currently runs too slow (need about 200 iter to reliably predict)

        # assert torch.allclose(predicted, test_y, atol=1e-2)


    # 4.2) Op Testing

    @ops(op_list=get_op_list([
        # Backend ops
        # ('sum',''),  # int dim list?
        ('add',''),
        ('sub',''),
        # ('rsub',''),
        ('div',''),
        ('mul',''),
        ('exp',''),
        # Decompositions
        # ('softmax',''),
        # ('nn.functional.mse_loss',''),
        # ('nn.functional.l1_loss',''),
    ]), allowed_dtypes=(torch.float,))
    @skipOps('NPTensorTest', 'test_np_tensor_parity', {
        # Need to accept `alpha` param
        xfail('add'),
        xfail('sub'),
        xfail('exp'),  # FIXME: This is weird - result off by 20 even when sigma is 1e-4!
        # We didn't implement a specific overload
        xfail('rsub',''),  # need: rsub.Tensor
    })
    def test_np_tensor_parity(self, device, dtype, op):
        # TODO: faster way to run tests, load all datasets at the same time?
        # Tests if tensor and subclass tensor compute the same values
        assert device == 'cpu' and  dtype == torch.float

        samples = list(op.sample_inputs(device, dtype, requires_grad=False))

        def supported_by_syft(sample):
            element_wise_binary_ops = "add sub rsub div mul".split(" ")
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            if any(t.numel() == 0 for t in arg_values if isinstance(t, torch.Tensor)) or \
                any(t.numel() == 0 for t in kwarg_values.values() if isinstance(t, torch.Tensor)):
                # FIXME: zero-numel tensor errors (data subject issue?)
                return False
            if any(t.ndim == 0 for t in arg_values if isinstance(t, torch.Tensor)) or \
                any(t.ndim == 0 for t in kwarg_values.values() if isinstance(t, torch.Tensor)):
                # FIXME: Scalar errors (data subject issue?)
                return False
            if op.name in element_wise_binary_ops and arg_values[0].shape[0] != arg_values[1].shape[0]:
                # FIXME: element-wise op when data_subjects_indexed arrays are differently shaped
                return False
            return True

        samples = list(filter(supported_by_syft, samples))
        wrapped_samples = self.wrap_samples_with_subclass(samples)
        for wrapped_sample, sample in zip(wrapped_samples, samples):
            # print("Sample: ", sample)
            # print("Wrapped sample: ", wrapped_sample)
            # Wrapped
            wrapped_args, wrapped_kwargs = wrapped_sample
            fn = op.get_op()
            raw_output = fn(*wrapped_args, **wrapped_kwargs)
            result = self.nptensorwrapper_to_torch(raw_output)

            # Non-Wrapped
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            expected = fn(*arg_values, **kwarg_values)
            print(result, expected, (result - expected).abs().max())
            # print(arg_values)

            self.assertTrue(_allclose_with_type_promotion(result, expected, rtol=1e-1, atol=1e-2))

    # @ops(op_list=get_op_list([
    #     # Backend ops
    #     ('sum',''),
    #     ('add',''),
    #     ('sub',''),
    #     ('div',''),
    #     ('mul',''),
    #     # Decompositions
    #     ('sigmoid',''),
    #     ('nn.functional.mse_loss',''),
    # ]), allowed_dtypes=(torch.float,))
    # @skipOps('NPTensorTest', 'test_np_tensor_gradients_parity', {
    #     # Forward is already failing for these
    #     xfail('add'),
    #     xfail('sub'),
    #     xfail('rsub',''),
    # })
    # def test_np_tensor_gradients_parity(self, device, dtype, op):
    #     # Tests if tensor and subclass tensor compute the same gradient
    #     assert device == 'cpu' and dtype == torch.float

    #     samples = op.sample_inputs(device, dtype, requires_grad=True)

    #     for sample in samples:
    #         arg_values = [sample.input] + list(sample.args)
    #         kwarg_values = sample.kwargs

    #         subclass_arg_values = tree_map(self.torch_to_nptensorwrapper, arg_values)
    #         subclass_kwarg_values = tree_map(self.torch_to_nptensorwrapper, kwarg_values)

    #         fn = op.get_op()

    #         subclass_out = fn(*subclass_arg_values, **subclass_kwarg_values)
    #         out = fn(*arg_values, **kwarg_values)

    #         assert isinstance(subclass_out, torch.Tensor), f"Expected op to return a single tensor, but got: {type(subclass_out)}"

    #         grad_outputs = torch.rand_like(out)
    #         subclass_grad_inputs = torch.autograd.grad(subclass_out, subclass_arg_values, grad_outputs=grad_outputs)
    #         subclass_grad_inputs = tree_map(self.nptensorwrapper_to_torch, subclass_grad_inputs)

    #         grad_inputs = torch.autograd.grad(out, arg_values, grad_outputs=grad_outputs)

    #         self.assertEqual(len(subclass_grad_inputs), len(grad_inputs))

    #         for subclass_grad_input, grad_input in zip(subclass_grad_inputs, grad_inputs):
    #             # print(subclass_grad_input, grad_input, (subclass_grad_input - grad_input).abs().max())
    #             # print(arg_values)
    #             self.assertTrue(torch.allclose(grad_input, subclass_grad_input))

instantiate_device_type_tests(NPTensorTest, globals(), only_for=("cpu,"))

if __name__ == "__main__":
    run_tests()