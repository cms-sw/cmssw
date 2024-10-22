#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"
#include <ostream>
#include "FWCore/Utilities/interface/Exception.h"

const AHCalDetId AHCalDetId::Undefined(0, 0, 0);

AHCalDetId::AHCalDetId() : DetId() {}

AHCalDetId::AHCalDetId(uint32_t rawid) : DetId(rawid) {}

AHCalDetId::AHCalDetId(int row, int col, int depth) : DetId(Hcal, HcalOther) {
  int icol = (col > 0) ? col : kMaxRowCol - col;
  int irow = (row > 0) ? row : kMaxRowCol - row;
  id_ |= ((depth & kHcalDepthMask) << HcalDetId::kHcalDepthOffset1) | (HcalDetId::kHcalZsideMask1) |
         ((irow & HcalDetId::kHcalEtaMask1) << HcalDetId::kHcalEtaOffset1) | (icol & HcalDetId::kHcalPhiMask1);
}

AHCalDetId::AHCalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet = (HcalSubdetector(gen.subdetId()));
    if (gen.det() != Hcal || subdet != HcalOther) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize AHCalDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

int AHCalDetId::irow() const {
  int value = ((id_ >> HcalDetId::kHcalEtaOffset1) & HcalDetId::kHcalEtaMask1);
  if (value >= kMaxRowCol)
    value = (kMaxRowCol - value);
  return value;
}

int AHCalDetId::icol() const {
  int value = (id_ & HcalDetId::kHcalPhiMask1);
  if (value >= kMaxRowCol)
    value = (kMaxRowCol - value);
  return value;
}

int AHCalDetId::depth() const { return ((id_ >> HcalDetId::kHcalDepthOffset1) & kHcalDepthMask); }

std::ostream& operator<<(std::ostream& s, const AHCalDetId& id) {
  return s << "(AHCal " << id.irow() << ',' << id.icol() << ',' << id.depth() << ')';
}
