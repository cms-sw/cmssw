///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPhoton.cc
// Description: Photons (generating single PE) in HF Shower Library
///////////////////////////////////////////////////////////////////////////////
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"

#include <iomanip>

HFShowerPhoton::HFShowerPhoton(float x, float y, float z, float lambda, float t)
    : position_(x, y, z), lambda_(lambda), time_(t) {}

HFShowerPhoton::HFShowerPhoton(const Point& p, float t, float lambda) : position_(p), lambda_(lambda), time_(t) {}

std::ostream& operator<<(std::ostream& os, const HFShowerPhoton& it) {
  os << "X " << std::setw(6) << it.x() << " Y " << std::setw(6) << it.y() << " Z " << std::setw(6) << it.z() << " t "
     << std::setw(6) << it.t() << " lambda " << it.lambda();
  return os;
}
