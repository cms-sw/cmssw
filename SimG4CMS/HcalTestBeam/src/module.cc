#include "SimG4CMS/HcalTestBeam/interface/HcalTB02SD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Analysis.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04Analysis.h"
#include "PluginManager/ModuleDef.h"
 
DEFINE_SEAL_MODULE ();
typedef HcalTB02SD HcalTB02SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalTB02SensitiveDetector);
DEFINE_SIMWATCHER (HcalTB02Analysis);
DEFINE_SIMWATCHER (HcalTB04Analysis);
