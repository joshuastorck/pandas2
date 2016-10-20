// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"
#include "pandas/types/common.h"
#include "pandas/types/numeric_fwd.h"

namespace pandas {

//
// Helper class that will define operator+ if the derived
// class defines operator+=
//
template <typename DERIVED>
class Addable {
 public:
  template <typename OTHER_TYPE>
  auto operator+(const OTHER_TYPE& other) const ->
      typename std::remove_reference<decltype(
          std::declval<DERIVED>() += std::declval<OTHER_TYPE>())>::type {
    DERIVED copy(*static_cast<const DERIVED*>(this));
    copy += other;
    return std::move(copy);
  }
};

//
// Same as Addable above.
//
template <typename DERIVED>
class Divisable {
 public:
  template <typename OTHER_TYPE>
  auto operator/(const OTHER_TYPE& other) const ->
      typename std::remove_reference<decltype(
          std::declval<DERIVED>() /= std::declval<OTHER_TYPE>())>::type {
    DERIVED copy(*static_cast<const DERIVED*>(this));
    copy /= other;
    return std::move(copy);
  }
};

template <typename TYPE>
class PANDAS_EXPORT NumericArray : public Array {
 public:
  using expected_base = NumericType<TYPE, typename TYPE::c_type, TYPE::type_id>;
  static_assert(std::is_assignable<expected_base, TYPE>::value,
      "NumericArray's type must be a numeric type");
  using T = typename TYPE::c_type;
  using DataTypePtr = std::shared_ptr<TYPE>;
  using Array::Array;

  NumericArray(const DataTypePtr& type, int64_t length, int64_t offset,
      const std::shared_ptr<Buffer>& data);

  // By making this final, the compiler can inline it when
  // we are invoking it through a reference to this class or
  // a sub-class, which happens in a number of the operators
  int64_t length() const final { return length_; }

  auto data() const -> const T*;
  auto mutable_data() const -> T*;

  TypePtr type() const override;

  // Performs copy-on-write semantics if needed, and sets
  // changed to true if it occurred.
  Status EnsureMutableAndCheckChange(bool& changed);

  // Despite being virtual, compiler could inline this if
  // the call is performed with a NumericArray reference
  const TYPE& type_reference() const override { return *type_; }

 protected:
  NumericArray(const NumericArray&);
  NumericArray(NumericArray&&);

  std::shared_ptr<TYPE> type_;
  int64_t length_;
  int64_t offset_;
  std::shared_ptr<Buffer> data_;
};

template <typename TYPE>
class PANDAS_EXPORT IntegerArray : public NumericArray<TYPE>,
                                   public Addable<IntegerArray<TYPE>> {
 public:
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data, int64_t offset = 0);
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits, int64_t offset = 0);

  IntegerArray(const IntegerArray& other);
  IntegerArray(IntegerArray&& other);

  int64_t GetNullCount() override;

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  bool owns_data() const override;

  Status EnsureMutable();

  // In the case where there are any nulls in this array,
  // this function will be faster than calling GetNullCount() > 0.
  bool HasNulls() const {
    for (auto ii = 0; ii < this->length_; ++ii) {
      if (IsNull(ii)) { return true; }
    }
    return false;
  }

  // Marks the value at the specified index as null. A pre-condition
  // is that the valid_bits for this object have already been
  // initialized.
  void SetNull(int64_t i) {
    // Pre-condition is that valid_bits_ is non-null because
    // operations that would set them (e.g. - operator+=)
    // would first create the valid_bits_ if needed
    BitUtil::SetBit(
        static_cast<MutableBuffer&>(*valid_bits_).mutable_data(), this->offset_ + i);
  }

  // Marks the value at the specified index as null. A pre-condition
  // is that the valid_bits for this object have already been
  // initialized.
  void SetValid(int64_t i) {
    // Pre-condition is that valid_bits_ is non-null because
    // operations that would set them (e.g. - operator+=)
    // would first create the valid_bits_ if needed
    BitUtil::ClearBit(
        static_cast<MutableBuffer&>(*valid_bits_).mutable_data(), this->offset_ + i);
  }

  // Returns whether or not the value at the specified index has
  // been marked as null.
  bool IsNull(int64_t i) const {
    if (valid_bits_) {
      // TODO: range check?
      return BitUtil::GetBit(valid_bits_->data(), this->offset_ + i);
    } else {
      return false;
    }
  }

  // Copy the list of nulls (aka valid bits) to a buffer
  Status CopyNulls(int64_t length, std::shared_ptr<Buffer>* out) const {
    // Pre-condition is that we already have valid_bits_ set
    return CopyBitmap(this->valid_bits_, this->offset_, length, out);
  }

  template <typename OTHER_TYPE>
  IntegerArray<TYPE>& operator+=(const IntegerArray<OTHER_TYPE>& other) {
    // TODO: check Status and throw exception
    EnsureMutable();
    auto length = std::min(this->length_, other.length());
    if (HasNulls()) {
      // Perform a |= on the valid bits if the other array has them.
      // TODO: do this more efficiently, perhaps by passing our
      //       valid_bits_ like this: other.BitwiseOr(*valid_bits_).
      //       It might also be useful to have a class that wraps
      //       much of the valid bit logic.
      if (other.HasNulls()) {
        for (auto ii = 0; ii < length; ++ii) {
          if (IsNull(ii) || other.IsNull(ii)) {
            SetNull(ii);
          } else {
            SetValid(ii);
          }
        }
      }
    } else if (other.HasNulls()) {
      std::shared_ptr<Buffer> new_valid_bits;
      // TODO: check Status and throw an exception
      other.CopyNulls(length, &new_valid_bits);
    }
    auto this_data = this->mutable_data();
    auto other_data = other.data();
    if (HasNulls()) {
      for (auto ii = 0; ii < length; ++ii, ++this_data, ++other_data) {
        if (!IsNull(ii)) { *this_data += *other_data; }
      }
    } else {
      // Can just copy directly
      // TODO: optimize with SIMD when TYPE::size == OTHER_TYPE::size
      for (auto ii = 0; ii < length; ++ii, ++this_data, ++other_data) {
        *this_data += *other_data;
      }
    }
    return *this;
  }

  // Compile time logic for determining the result type of
  // a division of two integers.
  template <typename OTHER_TYPE>
  class IntegerDivideInteger {
   public:
    static constexpr auto max_value =
        (std::numeric_limits<typename TYPE::c_type>::max() >
                    std::numeric_limits<typename OTHER_TYPE::c_type>::max()
                ? std::numeric_limits<typename TYPE::c_type>::max()
                : std::numeric_limits<typename OTHER_TYPE::c_type>::max());

    static constexpr auto needs_double =
        (max_value > std::numeric_limits<typename FloatType::c_type>::max());

    using type = typename std::conditional<needs_double, DoubleType, FloatType>::type;
  };

  template <typename OTHER_TYPE>
  FloatingArray<typename IntegerDivideInteger<OTHER_TYPE>::type> operator/(
      const IntegerArray<OTHER_TYPE>& other) const {
    using RESULT_TYPE = typename IntegerDivideInteger<OTHER_TYPE>::type;
    using CAST_TYPE = typename RESULT_TYPE::c_type;
    auto length = std::min(this->length_, other.length());
    auto data = std::make_shared<PoolBuffer>();
    // TODO: check Status and throw an exception
    data->Resize(length);
    FloatingArray<RESULT_TYPE> result(length, std::shared_ptr<Buffer>(std::move(data)));
    // TODO: use SIMD (Intel's _mm*_div_p{s,d} intrinsics) depending on the size
    //       of the resulting type
    auto this_data = this->data();
    auto other_data = other.data();
    // The result was just created and has a zero offset, so no need to add it
    auto result_data = result.mutable_data();
    for (int64_t ii = 0; ii < length; ++ii, ++this_data, ++other_data, ++result_data) {
      *result_data =
          static_cast<CAST_TYPE>(*this_data) / static_cast<CAST_TYPE>(*other_data);
    }
    return result;
  }

 private:
  std::shared_ptr<Buffer> valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArray : public NumericArray<TYPE>,
                                    public Addable<FloatingArray<TYPE>>,
                                    public Divisable<FloatingArray<TYPE>> {
 public:
  static_assert(std::is_floating_point<typename TYPE::c_type>::value,
      "Only floating point types are allowed for FloatingArray's DataType");

  FloatingArray(int64_t length, const std::shared_ptr<Buffer>& data, int64_t offset = 0);

  FloatingArray(const FloatingArray& other);
  FloatingArray(FloatingArray&& other);

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;
  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  int64_t GetNullCount() override;

  bool owns_data() const override;

  //
  // Arithmetic operators
  //

  template <template <typename> class ARRAY_TYPE, typename DATA_TYPE>
  FloatingArray<TYPE>& operator/=(const ARRAY_TYPE<DATA_TYPE>& other) {
    static auto divide_equals = [](
        typename TYPE::c_type& left, auto const& right) { left /= right; };
    return EvaluateBinaryOperator(divide_equals, other);
  }

  template <template <typename> class ARRAY_TYPE, typename DATA_TYPE>
  FloatingArray<TYPE>& operator+=(const ARRAY_TYPE<DATA_TYPE>& other) {
    static auto plus_equals = [](
        typename TYPE::c_type& left, auto const& right) { left += right; };
    return EvaluateBinaryOperator(plus_equals, other);
  }

 private:
  // Evaluates an arithmetic operator. If the right hand side is an IntegerArray,
  // this will evaluate whether or not the elements are null and then set
  // the results to NaN.
  template <typename OPERATOR, template <typename> class ARRAY_TYPE, typename DATA_TYPE>
  FloatingArray<TYPE>& EvaluateBinaryOperator(
      const OPERATOR& operation, const ARRAY_TYPE<DATA_TYPE>& other) {
    bool changed;
    this->EnsureMutableAndCheckChange(changed);
    auto added = std::min(this->length_, other.length());
    auto this_data = this->mutable_data();
    auto other_data = other.data();
    // TODO: use SIMD, special casing for the c_type of both
    //       the left and right hand sides
    for (auto ii = 0; ii < added; ++ii, ++this_data, ++other_data) {
      // Handle integer NaN's by using SFINAE on the type. If the right
      // hand side of this function is a FloatingArray, this will evaluate to
      // false at compile time
      if (IsNull(other, ii)) {
        *this_data = std::numeric_limits<typename TYPE::c_type>::quiet_NaN();
      } else {
        operation(*this_data, *other_data);
      }
    }
    return *this;
  }

  // SFINAE for IntegerArray
  template <template <typename> class ARRAY_TYPE, typename DATA_TYPE>
  typename std::enable_if<
      std::is_assignable<IntegerArray<DATA_TYPE>, ARRAY_TYPE<DATA_TYPE>>::value,
      bool>::type
  IsNull(const ARRAY_TYPE<DATA_TYPE>& other, int64_t i) {
    return other.IsNull(i);
  }

  // SFINAE for FloatingArray, compiler should remove any branches
  // that depend on this function
  template <template <typename> class ARRAY_TYPE, typename DATA_TYPE>
  constexpr typename std::enable_if<
      !std::is_assignable<IntegerArray<DATA_TYPE>, ARRAY_TYPE<DATA_TYPE>>::value,
      bool>::type
  IsNull(const ARRAY_TYPE<DATA_TYPE>& other, int64_t i) {
    return false;
  }
};

// Only instantiate these templates once
extern template class PANDAS_EXPORT IntegerArray<Int8Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt8Type>;
extern template class PANDAS_EXPORT IntegerArray<Int16Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt16Type>;
extern template class PANDAS_EXPORT IntegerArray<Int32Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt32Type>;
extern template class PANDAS_EXPORT IntegerArray<Int64Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt64Type>;
extern template class PANDAS_EXPORT FloatingArray<FloatType>;
extern template class PANDAS_EXPORT FloatingArray<DoubleType>;

extern template class PANDAS_EXPORT NumericArray<Int8Type>;
extern template class PANDAS_EXPORT NumericArray<UInt8Type>;
extern template class PANDAS_EXPORT NumericArray<Int16Type>;
extern template class PANDAS_EXPORT NumericArray<UInt16Type>;
extern template class PANDAS_EXPORT NumericArray<Int32Type>;
extern template class PANDAS_EXPORT NumericArray<UInt32Type>;
extern template class PANDAS_EXPORT NumericArray<Int64Type>;
extern template class PANDAS_EXPORT NumericArray<UInt64Type>;
extern template class PANDAS_EXPORT NumericArray<FloatType>;
extern template class PANDAS_EXPORT NumericArray<DoubleType>;

}  // namespace pandas
