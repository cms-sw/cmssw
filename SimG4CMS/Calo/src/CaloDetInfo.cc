#include "SimG4CMS/Calo/interface/CaloDetInfo.h"

#include <iomanip>

CaloDetInfo::CaloDetInfo(
    uint32_t id, uint32_t d, double r, const std::string& name, G4ThreeVector pos, const G4VSolid* solid, bool flag)
    : id_(id), depth_(d), rho_(r), name_(name), pos_(pos), solid_(solid), flag_(flag) {}

CaloDetInfo::CaloDetInfo()
    : id_(0), depth_(0), rho_(0), name_(""), pos_(G4ThreeVector(0, 0, 0)), solid_(nullptr), flag_(false) {}

CaloDetInfo::CaloDetInfo(const CaloDetInfo& right) {
  id_ = right.id_;
  depth_ = right.depth_;
  rho_ = right.rho_;
  name_ = right.name_;
  pos_ = right.pos_;
  solid_ = right.solid_;
  flag_ = right.flag_;
}

bool CaloDetInfo::operator<(const CaloDetInfo& info) const {
  if (id_ == info.id()) {
    if (depth_ == info.depth()) {
      return (rho_ > info.rho());
    } else {
      return (depth_ > info.depth());
    }
  } else {
    return (id_ > info.id());
  }
}

std::ostream& operator<<(std::ostream& os, const CaloDetInfo& info) {
  os << info.name() << " Id 0x" << std::hex << info.id() << std::dec << ":" << info.depth() << " R "
     << std::setprecision(4) << info.rho() << " Position " << info.pos();
  return os;
}
