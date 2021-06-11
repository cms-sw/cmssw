#ifndef SimG4Core_SensitiveDetector_sensitiveDetectorMakers_h
#define SimG4Core_SensitiveDetector_sensitiveDetectorMakers_h
// -*- C++ -*-
//
// Package:     SimG4Core/SensitiveDetector
// Class  :     sensitiveDetectorMakers
//
/**\function sensitiveDetectorMakers

 Description: Makes it easy to find what SensitiveDetectorMakerBase are available

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 07 Jun 2021 19:48:31 GMT
//

// system include files
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

// forward declarations
namespace edm {
  class ParameterSet;
}
namespace sim {
  std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> sensitiveDetectorMakers(
      edm::ParameterSet const&, edm::ConsumesCollector, std::vector<std::string> const& chosenMakers);
}
#endif
