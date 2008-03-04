#include "SimG4CMS/FP420/interface/FP420Test.h"
#include "SimG4CMS/FP420/interface/FP420SD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
//nclude "SimG4Core/Watcher/interface/SimWatcher.h" //=
//nclude "SimG4Core/Watcher/interface/SimWatcherMaker.h" //=
#include "FWCore/PluginManager/interface/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
typedef FP420SD FP420SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(FP420SensitiveDetector);
DEFINE_SIMWATCHER (FP420Test); //=


