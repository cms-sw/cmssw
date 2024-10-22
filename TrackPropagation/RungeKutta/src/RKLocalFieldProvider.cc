#include "RKLocalFieldProvider.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "FWCore/Utilities/interface/Likely.h"

RKLocalFieldProvider::RKLocalFieldProvider(const MagVolume& vol) : theVolume(vol), theFrame(vol), transform_(false) {}

RKLocalFieldProvider::RKLocalFieldProvider(const MagVolume& vol, const Frame& frame)
    : theVolume(vol), theFrame(frame), transform_(true) {}

RKLocalFieldProvider::Vector RKLocalFieldProvider::inTesla(const LocalPoint& lp) const {
  if UNLIKELY (transform_) {
    LocalPoint vlp(theVolume.toLocal(theFrame.toGlobal(lp)));
    return theFrame.toLocal(theVolume.toGlobal(theVolume.fieldInTesla(vlp))).basicVector();
  }
  return theVolume.fieldInTesla(lp).basicVector();
}
