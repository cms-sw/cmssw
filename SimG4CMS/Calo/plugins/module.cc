#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HGCSD.h"
#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"
#include "SimG4CMS/Calo/interface/HcalTestAnalysis.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
  

typedef ECalSD EcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(EcalSensitiveDetector);
typedef HCalSD HcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalSensitiveDetector);
typedef HGCSD HGCSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCSensitiveDetector);

DEFINE_SENSITIVEDETECTOR(CaloTrkProcessing);

DEFINE_SIMWATCHER (HcalTestAnalysis);
