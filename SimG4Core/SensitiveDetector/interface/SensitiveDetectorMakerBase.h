#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorMakerBase_h
#define SimG4Core_SensitiveDetector_SensitiveDetectorMakerBase_h
// -*- C++ -*-
//
// Package:     SensitiveDetector
// Class  :     SensitiveDetectorMakerBase
//
// Original Author:
//         Created:  Mon Nov 14 11:50:24 EST 2005
//

#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

// system include files
#include <string>

// forward declarations
class SimActivityRegistry;
class DDCompactView;
class SimTrackManager;
class SensitiveDetectorCatalog;

namespace edm {
  class ParameterSet;
}

class SensitiveDetectorMakerBase {
public:
  explicit SensitiveDetectorMakerBase(){};
  virtual ~SensitiveDetectorMakerBase(){};

  // ---------- const member functions ---------------------
  virtual SensitiveDetector* make(const std::string& iname,
                                  const DDCompactView& cpv,
                                  const SensitiveDetectorCatalog& clg,
                                  const edm::ParameterSet& p,
                                  const SimTrackManager* man,
                                  SimActivityRegistry& reg) const = 0;

private:
  SensitiveDetectorMakerBase(const SensitiveDetectorMakerBase&) = delete;
  const SensitiveDetectorMakerBase& operator=(const SensitiveDetectorMakerBase&) = delete;
};

#endif
