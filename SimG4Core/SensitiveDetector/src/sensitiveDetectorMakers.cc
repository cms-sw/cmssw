// -*- C++ -*-
//
// Package:     SimG4Core/SensitiveDetector
// Class  :     sensitiveDetectorMakers
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 07 Jun 2021 19:48:40 GMT
//

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/sensitiveDetectorMakers.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace sim {
  std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> sensitiveDetectorMakers(
      edm::ParameterSet const& pset, edm::ConsumesCollector cc, std::vector<std::string> const& chosenMakers) {
    std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> retValue;
    if (chosenMakers.empty()) {
      //load all
      auto const& categoriesToInfo = edmplugin::PluginManager::get()->categoryToInfos();
      auto infosItr = categoriesToInfo.find(SensitiveDetectorPluginFactory::get()->category());
      if (infosItr == categoriesToInfo.end()) {
        throw cms::Exception("MissingPlugins")
            << "When trying to load all SensitiveDetectorMakerBase, no plugins found";
      } else {
        for (auto const& info : infosItr->second) {
          retValue[info.name_] = SensitiveDetectorPluginFactory::get()->create(info.name_, pset, cc);
        }
      }
    } else {
      for (auto const& name : chosenMakers) {
        retValue[name] = SensitiveDetectorPluginFactory::get()->create(name, pset, cc);
      }
    }
    return retValue;
  }
}  // namespace sim
