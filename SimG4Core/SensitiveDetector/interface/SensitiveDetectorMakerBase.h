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
#include <memory>

// forward declarations
class SimActivityRegistry;
class SimTrackManager;
class SensitiveDetectorCatalog;

namespace edm {
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class SensitiveDetectorMakerBase {
public:
  explicit SensitiveDetectorMakerBase() = default;
  virtual ~SensitiveDetectorMakerBase();
  SensitiveDetectorMakerBase(const SensitiveDetectorMakerBase&) = delete;
  const SensitiveDetectorMakerBase& operator=(const SensitiveDetectorMakerBase&) = delete;

  virtual void beginRun(edm::EventSetup const&);

  // ---------- const member functions ---------------------
  //deprecated API
  virtual std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                                  const edm::EventSetup& es,
                                                  const SensitiveDetectorCatalog& clg,
                                                  const edm::ParameterSet& p,
                                                  const SimTrackManager* man,
                                                  SimActivityRegistry& reg) const;

  virtual std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                                  const SensitiveDetectorCatalog& clg,
                                                  const edm::ParameterSet& p,
                                                  const SimTrackManager* man,
                                                  SimActivityRegistry& reg) const;
};

#endif
