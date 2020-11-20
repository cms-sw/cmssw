#include "SimG4CMS/Calo/interface/CaloDetInfo.h"

#include <iomanip>

CaloDetInfo::CaloDetInfo(unsigned int id, const std::string& name, G4ThreeVector pos, const G4VSolid* solid, bool flag)
    : id_(id), name_(name), pos_(pos), solid_(solid), flag_(flag) {}

CaloDetInfo::CaloDetInfo() : id_(0), name_(""), pos_(G4ThreeVector(0, 0, 0)), solid_(nullptr), flag_(false) {}

CaloDetInfo::CaloDetInfo(const CaloDetInfo& right) {
  id_ = right.id_;
  name_ = right.name_;
  pos_ = right.pos_;
  solid_ = right.solid_;
  flag_ = right.flag_;
}

bool CaloDetInfo::operator<(const CaloDetInfo& info) const { return (id_ < info.id()) ? false : true; }

std::ostream& operator<<(std::ostream& os, const CaloDetInfo& info) {
  os << info.name() << " Id 0x" << std::hex << info.id() << std::dec << " Position " << std::setprecision(4)
     << info.pos();
  return os;
}
