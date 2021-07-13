#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HGCSD.h"
#include "SimG4CMS/Calo/interface/HGCalSD.h"
#include "SimG4CMS/Calo/interface/HGCScintSD.h"
#include "SimG4CMS/Calo/interface/HFNoseSD.h"
#include "SimG4CMS/Calo/interface/HcalTestAnalysis.h"
#include "SimG4CMS/Calo/interface/CaloSteppingAction.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

typedef ECalSD EcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(EcalSensitiveDetector);
typedef HCalSD HcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HcalSensitiveDetector);
typedef HGCSD HGCSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCSensitiveDetector);
typedef HGCalSD HGCalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCalSensitiveDetector);
typedef HGCScintSD HGCScintillatorSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCScintillatorSensitiveDetector);
typedef HFNoseSD HFNoseSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HFNoseSensitiveDetector);

DEFINE_SIMWATCHER(HcalTestAnalysis);
DEFINE_SIMWATCHER(CaloSteppingAction);
