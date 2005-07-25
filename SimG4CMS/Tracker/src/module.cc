#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (SensitiveDetectorPluginFactory, TkAccumulatingSensitiveDetector, "TkAccumulatingSensitiveDetector");
