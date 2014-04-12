#include "SimG4CMS/HcalTestBeam/interface/HcalTB02SD.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
 

typedef HcalTB02SD HcalTB02SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalTB02SensitiveDetector);
typedef HcalTB06BeamSD HcalTB06BeamDetector;
DEFINE_SENSITIVEDETECTOR(HcalTB06BeamDetector);
