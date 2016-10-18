// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/numeric.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/pytypes.h"
#include "pandas/type.h"
#include "pandas/type_traits.h"
#include "pandas/types/common.h"

namespace pandas {

// ----------------------------------------------------------------------
// Floating point base class

FloatingArray::FloatingArray(
    const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data)
    : NumericArray(type, length), data_(data) {}

// ----------------------------------------------------------------------
// Specific implementations

template <typename TYPE>
FloatingArrayImpl<TYPE>::FloatingArrayImpl(
    int64_t length, const std::shared_ptr<Buffer>& data)
    : FloatingArray(get_type_singleton<TYPE>(), length, data) {}

template <typename TYPE>
int64_t FloatingArrayImpl<TYPE>::GetNullCount() {
  // TODO(wesm)
  return 0;
}

template <typename TYPE>
PyObject* FloatingArrayImpl<TYPE>::GetItem(int64_t i) {
  return NULL;
}

template <typename TYPE>
Status FloatingArrayImpl<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;

  RETURN_NOT_OK(data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  *out = std::make_shared<FloatingArrayImpl<TYPE>>(length, copied_data);
  return Status::OK();
}

template <typename TYPE>
Status FloatingArrayImpl<TYPE>::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

template <typename TYPE>
const typename TYPE::c_type* FloatingArrayImpl<TYPE>::data() const {
  return reinterpret_cast<const T*>(data_->data());
}

template <typename TYPE>
typename TYPE::c_type* FloatingArrayImpl<TYPE>::mutable_data() const {
  auto mutable_buf = static_cast<MutableBuffer*>(data_.get());
  return reinterpret_cast<T*>(mutable_buf->mutable_data());
}

template <typename TYPE>
bool FloatingArrayImpl<TYPE>::owns_data() const {
  return data_.use_count() == 1;
}

// Instantiate templates
template class FloatingArrayImpl<FloatType>;
template class FloatingArrayImpl<DoubleType>;

// ----------------------------------------------------------------------
// Any integer

IntegerArray::IntegerArray(const TypePtr type, int64_t length,
    const std::shared_ptr<Buffer>& data, const std::shared_ptr<Buffer>& valid_bits)
    : NumericArray(type, length), data_(data), valid_bits_(valid_bits) {}

int64_t IntegerArray::GetNullCount() {
  // TODO(wesm)
  // return nulls_.set_count();
  return 0;
}

std::shared_ptr<Buffer> IntegerArray::data_buffer() const {
  return data_;
}

std::shared_ptr<Buffer> IntegerArray::valid_buffer() const {
  return valid_bits_;
}

// ----------------------------------------------------------------------
// Typed integers

template <typename TYPE>
IntegerArrayImpl<TYPE>::IntegerArrayImpl(int64_t length,
    const std::shared_ptr<Buffer>& data, const std::shared_ptr<Buffer>& valid_bits)
    : IntegerArray(get_type_singleton<TYPE>(), length, data, valid_bits) {}

template <typename TYPE>
const typename TYPE::c_type* IntegerArrayImpl<TYPE>::data() const {
  return reinterpret_cast<const T*>(data_->data());
}

template <typename TYPE>
typename TYPE::c_type* IntegerArrayImpl<TYPE>::mutable_data() const {
  auto mutable_buf = static_cast<MutableBuffer*>(data_.get());
  return reinterpret_cast<T*>(mutable_buf->mutable_data());
}

template <typename TYPE>
PyObject* IntegerArrayImpl<TYPE>::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  return PyLong_FromLongLong(data()[i]);
}

template <typename TYPE>
bool IntegerArrayImpl<TYPE>::owns_data() const {
  bool owns_data = data_.use_count() == 1;
  if (valid_bits_) { owns_data &= valid_bits_.use_count() == 1; }
  return owns_data;
}

template <typename TYPE>
Status IntegerArrayImpl<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  std::shared_ptr<Buffer> copied_valid_bits;

  RETURN_NOT_OK(data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  if (valid_bits_) {
    RETURN_NOT_OK(CopyBitmap(data_, offset, length, &copied_valid_bits));
  }
  *out = std::make_shared<FloatingArrayImpl<TYPE>>(length, copied_data);
  return Status::OK();
}

static Status PyObjectToInt64(PyObject* obj, int64_t* out) {
  PyObject* num = PyNumber_Long(obj);

  RETURN_IF_PYERROR();
  *out = PyLong_AsLongLong(num);
  Py_DECREF(num);
  return Status::OK();
}

template <typename TYPE>
Status IntegerArrayImpl<TYPE>::SetItem(int64_t i, PyObject* val) {
  if (!data_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Underlying buffer is immutable");
  }

  if (!valid_bits_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Valid bits buffer is immutable");
  }

  if (py::is_na(val)) {
    if (!valid_bits_) {
      // TODO: raise Python exception on error status
      RETURN_NOT_OK(AllocateValidityBitmap(length_, &valid_bits_));
    }
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    BitUtil::ClearBit(mutable_bits, i);
  } else {
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    if (valid_bits_) { BitUtil::SetBit(mutable_bits, i); }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}

// Instantiate templates
template class IntegerArrayImpl<UInt8Type>;
template class IntegerArrayImpl<Int8Type>;
template class IntegerArrayImpl<UInt16Type>;
template class IntegerArrayImpl<Int16Type>;
template class IntegerArrayImpl<UInt32Type>;
template class IntegerArrayImpl<Int32Type>;
template class IntegerArrayImpl<UInt64Type>;
template class IntegerArrayImpl<Int64Type>;

// ----------------------------------------------------------------------
// Implement Boolean as subclass of UInt8

BooleanArray::BooleanArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : UInt8Array(length, data, valid_bits) {
  type_ = kBoolean;
}

PyObject* BooleanArray::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  if (data()[i] > 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

Status BooleanArray::SetItem(int64_t i, PyObject* val) {
  if (!data_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Underlying buffer is immutable");
  }

  if (!valid_bits_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Valid bits buffer is immutable");
  }

  if (py::is_na(val)) {
    if (!valid_bits_) {
      // TODO: raise Python exception on error status
      RETURN_NOT_OK(AllocateValidityBitmap(length_, &valid_bits_));
    }
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    BitUtil::ClearBit(mutable_bits, i);
  } else {
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    if (valid_bits_) { BitUtil::SetBit(mutable_bits, i); }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}

}  // namespace pandas
