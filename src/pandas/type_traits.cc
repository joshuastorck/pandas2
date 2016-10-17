// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/type_traits.h"

namespace pandas {

#define MAKE_TYPE_SINGLETON(NAME) \
  const std::shared_ptr<DataType> k##NAME = std::make_shared<NAME##Type>();

MAKE_TYPE_SINGLETON(Boolean);
MAKE_TYPE_SINGLETON(Int8);
MAKE_TYPE_SINGLETON(UInt8);
MAKE_TYPE_SINGLETON(Int16);
MAKE_TYPE_SINGLETON(UInt16);
MAKE_TYPE_SINGLETON(Int32);
MAKE_TYPE_SINGLETON(UInt32);
MAKE_TYPE_SINGLETON(Int64);
MAKE_TYPE_SINGLETON(UInt64);
MAKE_TYPE_SINGLETON(Float);
MAKE_TYPE_SINGLETON(Double);
MAKE_TYPE_SINGLETON(PyObject);

}  // namespace pandas
