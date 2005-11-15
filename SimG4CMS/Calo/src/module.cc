#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
typedef ECalSD EcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(EcalSensitiveDetector);
typedef HCalSD HcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalSensitiveDetector);
// Add Hcal Stuff here
