#include "SimG4CMS/FP420/interface/FP420Test.h"
#include "SimG4CMS/FP420/interface/FP420SD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
  

typedef FP420SD FP420SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(FP420SensitiveDetector);
DEFINE_SIMWATCHER (FP420Test); //=


