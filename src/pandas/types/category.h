// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <sstream>
#include <string>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

struct CategoryType : public DataType {
  explicit CategoryType(const ArrayView& categories)
      : DataType(DataType::CATEGORY), categories_(categories) {}

  std::string ToString() const override;

  std::shared_ptr<const DataType> category_type() const {
    return categories_.data()->type();
  }

  const ArrayView& categories() const { return categories_; }

 protected:
  ArrayView categories_;
};

class CategoryArray : public Array {
 public:
  CategoryArray(ArrayView codes, const std::shared_ptr<CategoryType>& type);

  const ArrayView& codes() const { return codes_; }

  const ArrayView& categories() const { return type_->categories(); }

 private:
  ArrayView codes_;
  std::shared_ptr<CategoryType> type_;
};

}  // namespace pandas
