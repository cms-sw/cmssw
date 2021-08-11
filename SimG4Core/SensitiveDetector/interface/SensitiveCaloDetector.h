#ifndef SimG4Core_SensitiveDetector_SensitiveCaloDetector_H
#define SimG4Core_SensitiveDetector_SensitiveCaloDetector_H

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include <vector>
#include <string>

class SensitiveCaloDetector : public SensitiveDetector {
public:
  explicit SensitiveCaloDetector(const std::string& iname, const SensitiveDetectorCatalog& clg)
      : SensitiveDetector(iname, clg, true){};

  virtual void fillHits(edm::PCaloHitContainer&, const std::string& hname) = 0;
  virtual void reset(){};
};

#endif
