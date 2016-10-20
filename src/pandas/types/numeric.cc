// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/numeric.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/pytypes.h"
#include "pandas/type.h"
#include "pandas/types/common.h"

namespace pandas {

// ----------------------------------------------------------------------
// Generic numeric class

template <typename TYPE>
NumericArray<TYPE>::NumericArray(const DataTypePtr& type, int64_t length, int64_t offset,
    const std::shared_ptr<Buffer>& data)
    : type_(type), length_(length), offset_(offset), data_(data) {}

template <typename TYPE>
NumericArray<TYPE>::NumericArray(const NumericArray& other) = default;

template <typename TYPE>
NumericArray<TYPE>::NumericArray(NumericArray&& other) = default;

template <typename TYPE>
auto NumericArray<TYPE>::data() const -> const T* {
  return reinterpret_cast<const T*>(data_->data()) + offset_;
}

template <typename TYPE>
auto NumericArray<TYPE>::mutable_data() const -> T* {
  auto mutable_buf = static_cast<MutableBuffer*>(data_.get());
  return reinterpret_cast<T*>(mutable_buf->mutable_data()) + offset_;
}

template <typename TYPE>
TypePtr NumericArray<TYPE>::type() const {
  return type_;
}

template <typename TYPE>
Status NumericArray<TYPE>::EnsureMutableAndCheckChange(bool& changed) {
  if (this->data_.use_count() == 1) {
    changed = false;
    return Status::OK();
  }
  changed = true;
  std::shared_ptr<Buffer> new_data;
  RETURN_NOT_OK(this->data_->Copy(
      this->offset_, this->length_ * sizeof(typename TYPE::c_type), &new_data));
  this->data_ = new_data;
  this->offset_ = 0;
  return Status::OK();
}

// ----------------------------------------------------------------------
// Floating point class

template <typename TYPE>
FloatingArray<TYPE>::FloatingArray(
    int64_t length, const std::shared_ptr<Buffer>& data, int64_t offset)
    : NumericArray<TYPE>(TYPE::SINGLETON, length, offset, data) {}

template <typename TYPE>
FloatingArray<TYPE>::FloatingArray(const FloatingArray& other) = default;

template <typename TYPE>
FloatingArray<TYPE>::FloatingArray(FloatingArray&& other) = default;

template <typename TYPE>
int64_t FloatingArray<TYPE>::GetNullCount() {
  // TODO(wesm)
  return 0;
}

template <typename TYPE>
PyObject* FloatingArray<TYPE>::GetItem(int64_t i) {
  return NULL;
}

template <typename TYPE>
Status FloatingArray<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;

  RETURN_NOT_OK(this->data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  *out = std::make_shared<FloatingArray<TYPE>>(length, copied_data);
  return Status::OK();
}

template <typename TYPE>
Status FloatingArray<TYPE>::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

template <typename TYPE>
bool FloatingArray<TYPE>::owns_data() const {
  return this->data_.use_count() == 1;
}

// Instantiate templates
template class FloatingArray<FloatType>;
template class FloatingArray<DoubleType>;

template class NumericArray<FloatType>;
template class NumericArray<DoubleType>;

// ----------------------------------------------------------------------
// Typed integers

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(
    int64_t length, const std::shared_ptr<Buffer>& data, int64_t offset)
    : IntegerArray(length, data, nullptr, offset) {}

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits, int64_t offset)
    : NumericArray<TYPE>(TYPE::SINGLETON, length, offset, data),
      valid_bits_(valid_bits) {}

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(const IntegerArray& other) = default;

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(IntegerArray&& other) = default;

template <typename TYPE>
int64_t IntegerArray<TYPE>::GetNullCount() {
  // TODO(wesm)
  // return nulls_.set_count();
  return 0;
}

template <typename TYPE>
PyObject* IntegerArray<TYPE>::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  return PyLong_FromLongLong(this->data()[i]);
}

template <typename TYPE>
bool IntegerArray<TYPE>::owns_data() const {
  bool owns_data = this->data_.use_count() == 1;
  if (valid_bits_) { owns_data &= valid_bits_.use_count() == 1; }
  return owns_data;
}

template <typename TYPE>
Status IntegerArray<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  std::shared_ptr<Buffer> copied_valid_bits;

  RETURN_NOT_OK(this->data_->Copy(
      (this->offset_ + offset) * itemsize, length * itemsize, &copied_data));

  if (valid_bits_) {
    RETURN_NOT_OK(CopyBitmap(
        this->valid_bits_, this->offset_ + offset, length, &copied_valid_bits));
  }
  *out = std::make_shared<IntegerArray<TYPE>>(length, copied_data);
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
Status IntegerArray<TYPE>::SetItem(int64_t i, PyObject* val) {
  if (!this->data_->is_mutable()) {
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
      RETURN_NOT_OK(AllocateValidityBitmap(this->length_, &valid_bits_));
    }
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    BitUtil::ClearBit(mutable_bits, i);
  } else {
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    if (valid_bits_) { BitUtil::SetBit(mutable_bits, i); }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    this->mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}

template <typename TYPE>
Status IntegerArray<TYPE>::EnsureMutable() {
  bool changed;
  RETURN_NOT_OK(NumericArray<TYPE>::EnsureMutableAndCheckChange(changed));
  if (!changed) { return Status::OK(); }
  // Copy the valid bits
  if (valid_bits_) {
    std::shared_ptr<Buffer> new_valid_bits;
    RETURN_NOT_OK(
        CopyBitmap(this->valid_bits_, this->offset_, this->length_, &new_valid_bits));
    this->valid_bits_ = std::move(new_valid_bits);
  }
  return Status::OK();
}

// Instantiate templates
template class IntegerArray<UInt8Type>;
template class IntegerArray<Int8Type>;
template class IntegerArray<UInt16Type>;
template class IntegerArray<Int16Type>;
template class IntegerArray<UInt32Type>;
template class IntegerArray<Int32Type>;
template class IntegerArray<UInt64Type>;
template class IntegerArray<Int64Type>;

template class NumericArray<UInt8Type>;
template class NumericArray<Int8Type>;
template class NumericArray<UInt16Type>;
template class NumericArray<Int16Type>;
template class NumericArray<UInt32Type>;
template class NumericArray<Int32Type>;
template class NumericArray<UInt64Type>;
template class NumericArray<Int64Type>;

}  // namespace pandas
