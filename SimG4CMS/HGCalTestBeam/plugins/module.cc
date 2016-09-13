#include "SimG4CMS/HGCalTestBeam/interface/HGCalTB16SD01.h"
#include "SimG4CMS/HGCalTestBeam/interface/HGCalTBMB.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
typedef HGCalTB16SD01 HGCalTB1601SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCalTB1601SensitiveDetector);
DEFINE_SIMWATCHER (HGCalTBMB);

