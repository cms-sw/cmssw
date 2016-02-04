#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H
#define SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H

# include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
# include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMaker.h"
# include "FWCore/PluginManager/interface/PluginFactory.h"

#include <string>

typedef edmplugin::PluginFactory<SensitiveDetectorMakerBase *()> SensitiveDetectorPluginFactory;

#define DEFINE_SENSITIVEDETECTOR(type) \
  DEFINE_EDM_PLUGIN(SensitiveDetectorPluginFactory, SensitiveDetectorMaker<type>, #type)
#endif
