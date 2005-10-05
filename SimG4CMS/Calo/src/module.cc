#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (SensitiveDetectorPluginFactory, CaloSD, "CaloSD");
DEFINE_SEAL_PLUGIN (SensitiveDetectorPluginFactory, ECalSD, "EcalSensitiveDetector");
// Add Hcal Stuff here
