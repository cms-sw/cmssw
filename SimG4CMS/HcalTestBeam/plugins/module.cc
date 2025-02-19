#include "SimG4CMS/HcalTestBeam/interface/HcalTB02SD.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Analysis.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04Analysis.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Analysis.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
 

typedef HcalTB02SD HcalTB02SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalTB02SensitiveDetector);
typedef HcalTB06BeamSD HcalTB06BeamDetector;
DEFINE_SENSITIVEDETECTOR(HcalTB06BeamDetector);
DEFINE_SIMWATCHER (HcalTB02Analysis);
DEFINE_SIMWATCHER (HcalTB04Analysis);
DEFINE_SIMWATCHER (HcalTB06Analysis);
