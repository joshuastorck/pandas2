# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True

from pandas.internals.numpy cimport ndarray
cimport pandas.internals.numpy as cnp

from pandas.native cimport shared_ptr, string
cimport pandas.native as lp

from cpython cimport PyObject
from cython.operator cimport dereference as deref
cimport cpython

cdef extern from "Python.h":
    int PySlice_Check(object)

import pandas.internals.config
import numpy as np

UINT8 = lp.TypeId_UINT8
UINT16 = lp.TypeId_UINT16
UINT32 = lp.TypeId_UINT32
UINT64 = lp.TypeId_UINT64
INT8 = lp.TypeId_INT8
INT16 = lp.TypeId_INT16
INT32 = lp.TypeId_INT32
INT64 = lp.TypeId_INT64
BOOL = lp.TypeId_BOOL
FLOAT = lp.TypeId_FLOAT
DOUBLE = lp.TypeId_DOUBLE
PYOBJECT = lp.TypeId_PYOBJECT
CATEGORY = lp.TypeId_CATEGORY
TIMESTAMP = lp.TypeId_TIMESTAMP
TIMESTAMP_TZ = lp.TypeId_TIMESTAMP_TZ


class CPandasException(Exception):
    pass


class CPandasBadStatus(CPandasException):
    """
    libpandas operation returned an error Status
    """
    pass


class CPandasNotImplemented(CPandasBadStatus):
    pass


cdef check_status(const lp.Status& status):
    if status.ok():
        return

    message = status.ToString()

    if status.IsNotImplemented():
        raise CPandasNotImplemented(message)
    else:
        raise CPandasBadStatus(message)


cdef class Scalar:
    cdef readonly:
        lp.TypeId type


cdef class NAType(Scalar):

    def __cinit__(self):
        self.type = lp.TypeId_NA

    def __repr__(self):
        return 'NA'

NA = NAType()
lp.init_natype(NAType, NA)

def isnull(obj):
    return lp.is_na(obj)


cdef dict _primitive_type_aliases = {
    'u1': UINT8,
    'u2': UINT16,
    'u4': UINT32,
    'u8': UINT64,
    'uint8': UINT8,
    'uint16': UINT16,
    'uint32': UINT32,
    'uint64': UINT64,

    'i1': INT8,
    'i2': INT16,
    'i4': INT32,
    'i8': INT64,
    'int8': INT8,
    'int16': INT16,
    'int32': INT32,
    'int64': INT64,

    'f4': FLOAT,
    'f8': DOUBLE,
    'float32': FLOAT,
    'float64': DOUBLE,

    'b1': BOOL,
    'bool': BOOL,

    'O8': PYOBJECT,
    'object': PYOBJECT,
}


def wrap_numpy_array(ndarray arr):
    pass


cdef class PandasType:
    cdef:
        lp.TypePtr type

    cdef init(self, const lp.TypePtr& type):
        self.type = type

    def __repr__(self):
        return 'PandasType({0})'.format(self.name)

    property name:

        def __get__(self):
            return self.type.get().ToString()

    def equals(PandasType self, PandasType other):
        return self.type.get().Equals(deref(other.type.get()))


def primitive_type(TypeId tp_enum):
    cdef:
        lp.TypePtr sp_type
        lp.DataType* type

    check_status(lp.primitive_type_from_enum(tp_enum, &type))
    sp_type.reset(type)
    return wrap_type(sp_type)


cdef class Category(PandasType):

    property categories:

        def __get__(self):
            pass


def category_type(categories):
    pass


cdef class Array:
    cdef:
        lp.ArrayPtr arr
        CArray* ap

    def __cinit__(self):
        self.ap = NULL

    cdef init(self, const lp.ArrayPtr& arr):
        self.arr = arr
        self.ap = arr.get()

    def __len__(self):
        return self.ap.length()

    property dtype:

        def __get__(self):
            return wrap_type(self.ap.type())

    # def to_numpy(self):
    #     return self.ap.to_numpy()

    def __getitem__(self, i):
        cdef:
            Py_ssize_t n = len(self)

        if PySlice_Check(i):
            start = i.start or 0
            while start < 0:
                start += n

            stop = i.stop if i.stop is not None else n
            while stop < 0:
                stop += n

            step = i.step or 1
            if step != 1:
                raise NotImplementedError
            else:
                return self.slice(start, stop)

        while i < 0:
            i += self.ap.length()

        return self._getitem(i)

    cdef inline _getitem(self, size_t i):
        if i >= self.ap.length():
            raise IndexError('Out of bounds: %d' % i)
        return self.ap.GetItem(i)

    def __setitem__(self, i, val):
        cdef:
            Py_ssize_t n = len(self)

        if PySlice_Check(i):
            raise NotImplementedError

        while i < 0:
            i += self.ap.length()

        self._setitem(i, val)

    cdef inline _setitem(self, size_t i, object val):
        if i >= self.ap.length():
            raise IndexError('Out of bounds: %d' % i)
        self.ap.SetItem(i, val)

    def slice(self, start, end):
        pass



cdef class NumericArray(Array):
    pass


cdef class FloatingArray(NumericArray):
    pass


cdef class IntegerArray(NumericArray):
    pass


cdef class Float32Array(FloatingArray):
    pass


cdef class BooleanArray(Array):
    cdef:
        lp.CBooleanArray* inst

    cdef init(self, const ArrayPtr& arr):
        Array.init(self, arr)


cdef class CategoryArray(Array):
    pass


cdef Array wrap_array(const lp.ArrayPtr& arr):
    cdef:
        Array result

    if arr.get().type_id() == lp.TypeId_CATEGORY:
        result = CategoryArray()
    else:
        result = Array()

    result.init(arr)

    return result


cdef PandasType wrap_type(const lp.TypePtr& sp_type):
    cdef:
        lp.DataType* type = sp_type.get()
        PandasType result

    if type.type() == lp.TypeId_CATEGORY:
        result = Category()
    else:
        result = PandasType()

    result.init(sp_type)

    return result


cpdef PandasType convert_numpy_dtype(cnp.dtype dtype):
    cdef lp.TypeId pandas_typenum
    check_status(lp.numpy_type_num_to_pandas(dtype.type_num,
                                             &pandas_typenum))

    return primitive_type(pandas_typenum)


def to_array(values):
    if isinstance(values, np.ndarray):
        return numpy_to_pandas_array(values)
    else:
        raise TypeError(type(values))


cdef numpy_to_pandas_array(ndarray arr):
    cdef:
        Array result
        CArray* array_obj
        lp.ArrayPtr sp_array

    check_status(lp.array_from_numpy(<PyObject*> arr, &array_obj))
    sp_array.reset(array_obj)
    return wrap_array(sp_array)
