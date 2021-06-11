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
#include "FWCore/Framework/interface/ConsumesCollector.h"

// forward declarations
class SimTrackManager;
class SimActivityRegistry;
class SensitiveDetectorCatalog;

namespace edm {
  class EventSetup;
  class ParameterSet;
}  // namespace edm

template <class T>
class SensitiveDetectorMaker : public SensitiveDetectorMakerBase {
public:
  explicit SensitiveDetectorMaker(edm::ParameterSet const&, edm::ConsumesCollector){};
  SensitiveDetectorMaker(const SensitiveDetectorMaker&) = delete;
  const SensitiveDetectorMaker& operator=(const SensitiveDetectorMaker&) = delete;

  // ---------- const member functions ---------------------
  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const edm::EventSetup& es,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const override {
    auto sd = std::make_unique<T>(iname, es, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  };
};

#endif
