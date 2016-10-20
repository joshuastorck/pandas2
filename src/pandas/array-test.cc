// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Test non-type specific array functionality

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/meta/typelist.h"
#include "pandas/test-util.h"
#include "pandas/type.h"
#include "pandas/types/numeric.h"

using std::string;

namespace pandas {

class TestArray : public ::testing::Test {
 public:
  void SetUp() {
    values_ = {0, 1, 2, 3, 4, 5, 6, 7};

    auto buffer =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(values_.data()),
            values_.size() * sizeof(double));

    array_ = std::make_shared<DoubleArray>(values_.size(), buffer);
  }

 protected:
  std::shared_ptr<Array> array_;
  std::vector<double> values_;
};

TEST_F(TestArray, Attrs) {
  DoubleType ex_type;
  ASSERT_TRUE(array_->type()->Equals(ex_type));
  ASSERT_EQ(DataType::FLOAT64, array_->type_id());

  ASSERT_EQ(values_.size(), array_->length());
}

template <typename LEFT_ARRAY_TYPE, typename RIGHT_ARRAY_TYPE, std::size_t LENGTH = 10>
class OperatorTestData {
 public:
  OperatorTestData()
      : left_buffer_(std::make_shared<Buffer>(
            reinterpret_cast<const std::uint8_t*>(Initialize(left_data_)),
            LENGTH * sizeof(LEFT_C_TYPE))),
        right_buffer_(std::make_shared<Buffer>(
            reinterpret_cast<const std::uint8_t*>(Initialize(right_data_)),
            LENGTH * sizeof(RIGHT_C_TYPE))),
        left_array_(LENGTH, left_buffer_),
        right_array_(LENGTH, right_buffer_) {}

  template <typename C_TYPE>
  static C_TYPE* Initialize(C_TYPE (&value)[LENGTH]) {
    for (auto ii = 0; ii < LENGTH; ++ii) {
      // Start at 1 so that we don't get FPE with operator/
      value[ii] = static_cast<C_TYPE>(ii + 1);
    }
    return value;
  }

  using LEFT_C_TYPE = typename LEFT_ARRAY_TYPE::T;

  using RIGHT_C_TYPE = typename RIGHT_ARRAY_TYPE::T;

  LEFT_C_TYPE left_data_[LENGTH];

  RIGHT_C_TYPE right_data_[LENGTH];

  std::shared_ptr<Buffer> left_buffer_;

  std::shared_ptr<Buffer> right_buffer_;

  LEFT_ARRAY_TYPE left_array_;

  RIGHT_ARRAY_TYPE right_array_;
};

template <typename OPERATOR, typename INPLACE_OPERATOR, std::size_t LENGTH = 10>
class TestInplaceOperator {
 public:
  TestInplaceOperator(OPERATOR& operation, INPLACE_OPERATOR& inplace_operation)
      : operation_(operation), inplace_operation_(inplace_operation) {}

  template <typename PAIR>
  void operator()() {
    using FIRST = typename std::tuple_element<0, PAIR>::type;
    using SECOND = typename std::tuple_element<1, PAIR>::type;
    OperatorTestData<FIRST, SECOND> test_data;
    auto result = operation_(test_data.left_array_, test_data.right_array_);
    for (auto ii = 0; ii < test_data.left_array_.length(); ++ii) {
      ASSERT_EQ(result.data()[ii],
          operation_(test_data.left_data_[ii], test_data.right_data_[ii]));
    }
    inplace_operation_(test_data.left_array_, test_data.right_array_);
    for (auto ii = 0; ii < test_data.left_array_.length(); ++ii) {
      ASSERT_EQ(test_data.left_array_.data()[ii],
          operation_(test_data.left_data_[ii], test_data.right_data_[ii]));
    }
    for (auto ii = 0; ii < test_data.left_array_.length(); ++ii) {
      ASSERT_EQ(test_data.left_array_.data()[ii], result.data()[ii]);
    }
  }

 private:
  OPERATOR& operation_;

  INPLACE_OPERATOR inplace_operation_;
};

template <typename OPERATOR, std::size_t LENGTH = 10>
class TestOperator {
 public:
  TestOperator(OPERATOR& operation) : operation_(operation) {}

  template <typename PAIR>
  void operator()() {
    using FIRST = typename std::tuple_element<0, PAIR>::type;
    using SECOND = typename std::tuple_element<1, PAIR>::type;
    OperatorTestData<FIRST, SECOND> test_data;
    auto result = operation_(test_data.left_array_, test_data.right_array_);
    for (auto ii = 0; ii < test_data.left_array_.length(); ++ii) {
      ASSERT_EQ(result.data()[ii],
          operation_(test_data.left_data_[ii], test_data.right_data_[ii]));
    }
  }

 private:
  OPERATOR& operation_;
};

using IntegerTypes = TypeList<IntegerArray<UInt8Type>, IntegerArray<UInt16Type>,
    IntegerArray<UInt32Type>, IntegerArray<UInt64Type>, IntegerArray<Int8Type>,
    IntegerArray<Int16Type>, IntegerArray<Int32Type>, IntegerArray<Int64Type>>;

using FloatingPointTypes = TypeList<FloatingArray<FloatType>, FloatingArray<DoubleType>>;

using NumericTypes = decltype(IntegerTypes() + FloatingPointTypes());

TEST(TestArrayOperators, Addition) {
  auto plus = [](auto const& left, auto const& right) { return left + right; };
  auto plus_inplace = [](auto& left, auto const& right) { left += right; };

  static constexpr auto addition_tests =
      FloatingPointTypes().CartesianProduct(NumericTypes());
  TestInplaceOperator<decltype(plus), decltype(plus_inplace)> tester(plus, plus_inplace);
  addition_tests.Iterate(tester);
}

TEST(TestArrayOperators, Division) {
  auto divide = [](auto const& left, auto const& right) { return left / right; };
  auto divide_inplace = [](auto& left, auto const& right) { left /= right; };

  TestOperator<decltype(divide)> test(divide);
  IntegerTypes().CartesianProduct(IntegerTypes()).Iterate(test);

  TestInplaceOperator<decltype(divide), decltype(divide_inplace)> inplace_test(
      divide, divide_inplace);
  FloatingPointTypes().CartesianProduct(NumericTypes()).Iterate(inplace_test);
}

// ----------------------------------------------------------------------
// Array view object

class TestArrayView : public ::testing::Test {
 public:
  using value_t = double;

  void SetUp() {
    values_ = {0, 1, 2, 3, 4, 5, 6, 7};

    auto buffer =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(values_.data()),
            values_.size() * sizeof(value_t));

    auto arr = std::make_shared<DoubleArray>(values_.size(), buffer);
    view_ = ArrayView(arr);
  }

 protected:
  ArrayView view_;
  std::vector<value_t> values_;
};

TEST_F(TestArrayView, Ctors) {
  ASSERT_EQ(1, view_.ref_count());
  ASSERT_EQ(0, view_.offset());
  ASSERT_EQ(values_.size(), view_.length());

  // Copy ctor
  ArrayView view2(view_);
  ASSERT_EQ(2, view2.ref_count());
  ASSERT_EQ(0, view_.offset());
  ASSERT_EQ(values_.size(), view_.length());

  // move ctor
  ArrayView view3(view_.data(), 3);
  ArrayView view4(std::move(view3));
  ASSERT_EQ(3, view4.ref_count());
  ASSERT_EQ(3, view3.offset());
  ASSERT_EQ(values_.size() - 3, view3.length());

  // With offset and length
  ArrayView view5(view4.data(), 2, 4);
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());

  // Copy assignment
  ArrayView view6 = view5;
  ASSERT_EQ(5, view4.ref_count());
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());

  // Move assignment
  ArrayView view7 = std::move(view6);
  ASSERT_EQ(5, view4.ref_count());
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());
}

TEST_F(TestArrayView, EnsureMutable) {
  // This only tests for one data type -- we will need to test more rigorously
  // across all data types elsewhere

  const Array* ap = view_.data().get();

  ASSERT_OK(view_.EnsureMutable());
  ASSERT_EQ(ap, view_.data().get());

  ArrayView view2 = view_;

  ASSERT_OK(view_.EnsureMutable());

  // The views now have their own distinct copies of the array
  ASSERT_NE(ap, view_.data().get());
  ASSERT_EQ(ap, view2.data().get());

  ASSERT_EQ(1, view_.ref_count());
  ASSERT_EQ(1, view2.ref_count());
}

TEST_F(TestArrayView, Slice) {
  ArrayView s1 = view_.Slice(3);
  ASSERT_EQ(2, s1.ref_count());
  ASSERT_EQ(3, s1.offset());
  ASSERT_EQ(view_.length() - 3, s1.length());

  ArrayView s2 = view_.Slice(2, 4);
  ASSERT_EQ(3, s2.ref_count());
  ASSERT_EQ(2, s2.offset());
  ASSERT_EQ(4, s2.length());

  // Slice of a slice
  ArrayView s3 = s1.Slice(2);
  ASSERT_EQ(4, s3.ref_count());
  ASSERT_EQ(5, s3.offset());
  ASSERT_EQ(view_.length() - 5, s3.length());

  ArrayView s4 = s1.Slice(1, 2);
  ASSERT_EQ(5, s4.ref_count());
  ASSERT_EQ(4, s4.offset());
  ASSERT_EQ(2, s4.length());
}

}  // namespace pandas
