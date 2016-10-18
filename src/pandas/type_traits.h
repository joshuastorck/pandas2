// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/numpy_interop.h"
#include "pandas/types/numeric.h"
#include "pandas/types/pyobject.h"

namespace pandas {

template <int NPY_TYPE>
struct NumPyTraits {};

#define NUMPY_TRAITS_DECL(NPY_TYPE, PandasArrayType) \
  template <>                                        \
  struct NumPyTraits<NPY_TYPE> {                     \
    using ArrayType = PandasArrayType;               \
    using T = typename PandasArrayType::T;           \
  }

NUMPY_TRAITS_DECL(NPY_INT8, Int8Array);
NUMPY_TRAITS_DECL(NPY_INT16, Int16Array);
NUMPY_TRAITS_DECL(NPY_INT32, Int32Array);
NUMPY_TRAITS_DECL(NPY_INT64, Int64Array);
NUMPY_TRAITS_DECL(NPY_UINT8, UInt8Array);
NUMPY_TRAITS_DECL(NPY_UINT16, UInt16Array);
NUMPY_TRAITS_DECL(NPY_UINT32, UInt32Array);
NUMPY_TRAITS_DECL(NPY_UINT64, UInt64Array);
NUMPY_TRAITS_DECL(NPY_FLOAT32, FloatArray);
NUMPY_TRAITS_DECL(NPY_FLOAT64, DoubleArray);
NUMPY_TRAITS_DECL(NPY_OBJECT, PyObjectArray);
NUMPY_TRAITS_DECL(NPY_BOOL, BooleanArray);

template <typename T>
static inline std::shared_ptr<DataType> get_type_singleton() {
  return nullptr;
}

#define DECLARE_TYPE_SINGLETON(NAME)                                  \
  extern PANDAS_EXPORT const std::shared_ptr<DataType> k##NAME;       \
  template <>                                                         \
  inline std::shared_ptr<DataType> get_type_singleton<NAME##Type>() { \
    return k##NAME;                                                   \
  }

DECLARE_TYPE_SINGLETON(Boolean);
DECLARE_TYPE_SINGLETON(Int8);
DECLARE_TYPE_SINGLETON(UInt8);
DECLARE_TYPE_SINGLETON(Int16);
DECLARE_TYPE_SINGLETON(UInt16);
DECLARE_TYPE_SINGLETON(Int32);
DECLARE_TYPE_SINGLETON(UInt32);
DECLARE_TYPE_SINGLETON(Int64);
DECLARE_TYPE_SINGLETON(UInt64);
DECLARE_TYPE_SINGLETON(Float);
DECLARE_TYPE_SINGLETON(Double);
DECLARE_TYPE_SINGLETON(PyObject);

}  // namespace pandas
