import tvm
import torch
import numpy as np

def cast_array(array):
    if isinstance(array,tvm.runtime.ndarray.NDArray):
        array=array.asnumpy()
    elif isinstance(array,torch.Tensor):
        array=array.detach().cpu().numpy()
    assert isinstance(array,np.ndarray),"Only accept array as numpy.ndarray, get "+str(type(array))
    return array

def array_des(array):
    type_des=array.__class__.__name__
    array=cast_array(array)
    return "<{}>[{};{}] max {:g}, min {:g}, sum {:g}".format(
        type_des,','.join([str(s) for s in array.shape]),array.dtype.name,
        array.max(),array.min(),array.sum())

def array_compare(arrayA,arrayB,nameA="A",nameB="B",error=0.05):
    arrayA=cast_array(arrayA)
    arrayB=cast_array(arrayB)
    if arrayA.dtype!=arrayB.dtype:
        print("dtype mismatch between {} and {}".format(arrayA.dtype,arrayB.dtype))
    if arrayA.shape!=arrayB.shape:
        print("dtype mismatch between {} and {}".format(arrayA.dtype,arrayB.dtype))
    diff=(arrayA-arrayB)/(abs(arrayA)+0.0001)
    msg="max : {:g}, min :{:g}, sum : {:g}".format(diff.max(),diff.min(),diff.sum())
    if abs(diff).max()>error:
        print("[FAIL] "+msg)
        print("{} : {}".format(nameA,array_des(arrayA)))
        print("{} : {}".format(nameB,array_des(arrayB)))
        return False
    print("[PASS] "+msg)
    return True