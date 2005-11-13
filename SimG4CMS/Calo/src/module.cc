#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (SensitiveDetectorPluginFactory, ECalSD, "EcalSensitiveDetector");
DEFINE_SEAL_PLUGIN (SensitiveDetectorPluginFactory, HCalSD, "HcalSensitiveDetector");

