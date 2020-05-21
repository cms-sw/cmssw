#include <utility>

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

MaterialBudgetFormat::MaterialBudgetFormat(std::shared_ptr<MaterialBudgetData> data) : theData(std::move(data)) {}
