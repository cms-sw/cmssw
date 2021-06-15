#ifndef SimG4Core_SensitiveDetector_SensitiveTkDetector_H
#define SimG4Core_SensitiveDetector_SensitiveTkDetector_H

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include <string>

class SensitiveTkDetector : public SensitiveDetector {
public:
  explicit SensitiveTkDetector(const std::string& iname, const SensitiveDetectorCatalog& clg)
      : SensitiveDetector(iname, clg, false) {}
  virtual void fillHits(edm::PSimHitContainer&, const std::string& hname) = 0;
};

#endif
