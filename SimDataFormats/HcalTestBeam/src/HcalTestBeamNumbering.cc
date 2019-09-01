#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include <iostream>

//#define EDM_ML_DEBUG

uint32_t HcalTestBeamNumbering::packIndex(int det, int lay, int x, int y) {
  int ix(0), ixx(x);
  if (x < 0) {
    ix = 1;
    ixx = -x;
  }
  int iy(0), iyy(y);
  if (y < 0) {
    iy = 1;
    iyy = -y;
  }
  uint32_t idx = (det & kHcalBeamTypeMask) << kHcalBeamTypeOffset;
  idx += (lay & kHcalBeamLayerMask) << kHcalBeamLayerOffset;
  idx += (iy & kHcalBeamYSignMask) << kHcalBeamYSignOffset;
  idx += (iyy & kHcalBeamYValueMask) << kHcalBeamYValueOffset;
  idx += (ix & kHcalBeamXSignMask) << kHcalBeamXSignOffset;
  idx += (ixx & kHcalBeamXValueMask) << kHcalBeamXValueOffset;

#ifdef EDM_ML_DEBUG
  std::cout << "HcalTestBeamNumbering: Detector " << det << " Layer " << lay << " x " << x << " " << ix << " " << ixx
            << " y " << y << " " << iy << " " << iyy << " ID " << std::hex << idx << std::dec << std::endl;
#endif
  return idx;
}

void HcalTestBeamNumbering::unpackIndex(const uint32_t& idx, int& det, int& lay, int& x, int& y) {
  det = (idx >> kHcalBeamTypeOffset) & kHcalBeamTypeMask;
  lay = (idx >> kHcalBeamLayerOffset) & kHcalBeamLayerMask;
  y = (idx >> kHcalBeamYValueOffset) & kHcalBeamYValueMask;
  x = (idx >> kHcalBeamXValueOffset) & kHcalBeamXValueMask;
  if (((idx >> kHcalBeamYSignOffset) & kHcalBeamYSignMask) == 1)
    y = -y;
  if (((idx >> kHcalBeamXSignOffset) & kHcalBeamXSignMask) == 1)
    x = -x;

#ifdef EDM_ML_DEBUG
  std::cout << "HcalTestBeamNumbering: ID " << std::hex << idx << std::dec << " Detector " << det << " Layer " << lay
            << " x " << x << " y " << y << std::endl;
#endif
}
