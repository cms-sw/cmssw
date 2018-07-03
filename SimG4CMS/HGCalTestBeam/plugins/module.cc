#include "SimG4CMS/HGCalTestBeam/interface/HGCalTB16SD01.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalSD.h"
#include "SimG4CMS/HGCalTestBeam/interface/HGCalTBMB.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
typedef HGCalTB16SD01 HGCalTB1601SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCalTB1601SensitiveDetector);
typedef AHCalSD AHcalSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(AHcalSensitiveDetector);

DEFINE_SIMWATCHER (HGCalTBMB);

