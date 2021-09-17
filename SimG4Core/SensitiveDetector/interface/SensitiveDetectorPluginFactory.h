#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H
#define SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H

#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMaker.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

namespace edm {
  class ParameterSet;
}
typedef edmplugin::PluginFactory<SensitiveDetectorMakerBase *(edm::ParameterSet const &, edm::ConsumesCollector)>
    SensitiveDetectorPluginFactory;

#define DEFINE_SENSITIVEDETECTOR(type) \
  DEFINE_EDM_PLUGIN(SensitiveDetectorPluginFactory, SensitiveDetectorMaker<type>, #type)

#define DEFINE_SENSITIVEDETECTORBUILDER(type, name) DEFINE_EDM_PLUGIN(SensitiveDetectorPluginFactory, type, #name)

#endif
