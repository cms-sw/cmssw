#include <array>

namespace GEMDetLabel {
  static const std::array<std::string, 4> l_suffix = {{"_l1", "_l2", "_l1or2", "_l1and2"}};
  static const std::array<std::string, 2> s_suffix = {{"_st1", "_st2"}};
  static const std::array<std::string, 3> c_suffix = {{"_all", "_odd", "_even"}};
}  // namespace GEMDetLabel
