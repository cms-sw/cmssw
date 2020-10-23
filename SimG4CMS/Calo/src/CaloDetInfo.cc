#include "SimG4CMS/Calo/interface/CaloDetInfo.h"

#include <iomanip>

//#define EDM_ML_DEBUG

CaloDetInfo::CaloDetInfo(unsigned int id, std::string name, G4ThreeVector pos, std::vector<double> par)
    : id_(id), name_(name), pos_(pos), par_(par) {}

CaloDetInfo::CaloDetInfo() : id_(0), name_(""), pos_(G4ThreeVector(0, 0, 0)) {}

CaloDetInfo::CaloDetInfo(const CaloDetInfo& right) {
  id_ = right.id_;
  name_ = right.name_;
  pos_ = right.pos_;
  par_ = right.par_;
}

bool CaloDetInfo::operator<(const CaloDetInfo& info) const { return (id_ < info.id()) ? false : true; }

std::ostream& operator<<(std::ostream& os, const CaloDetInfo& info) {
  os << info.name() << " Id 0x" << std::hex << info.id() << std::dec << " Position " << std::setprecision(4)
     << info.pos() << " with " << info.par().size() << " parameters:";
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < info.par().size(); ++k)
    os << " [" << k << "] " << std::setprecision(5) << info.par()[k];
#endif
  return os;
}
