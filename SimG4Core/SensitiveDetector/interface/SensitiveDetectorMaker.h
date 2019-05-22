#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorMaker_h
#define SimG4Core_SensitiveDetector_SensitiveDetectorMaker_h
// -*- C++ -*-
//
// Package:     SensitiveDetector
// Class  :     SensitiveDetectorMaker
//
//
// Original Author:
//         Created:  Mon Nov 14 11:56:05 EST 2005
//

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations
class DDCompactView;
class SimTrackManager;
class SimActivityRegistry;
class SensitiveDetectorCatalog;

namespace edm {
  class ParameterSet;
}

template <class T>
class SensitiveDetectorMaker : public SensitiveDetectorMakerBase {
public:
  explicit SensitiveDetectorMaker(){};

  // ---------- const member functions ---------------------
  SensitiveDetector* make(const std::string& iname,
                          const DDCompactView& cpv,
                          const SensitiveDetectorCatalog& clg,
                          const edm::ParameterSet& p,
                          const SimTrackManager* man,
                          SimActivityRegistry& reg) const override {
    T* sd = new T(iname, cpv, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd);
    return static_cast<SensitiveDetector*>(sd);
  };

private:
  SensitiveDetectorMaker(const SensitiveDetectorMaker&) = delete;
  const SensitiveDetectorMaker& operator=(const SensitiveDetectorMaker&) = delete;
};

#endif
