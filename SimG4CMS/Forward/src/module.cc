#include "SimG4CMS/Forward/interface/CastorSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
typedef CastorSD CastorSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CastorSensitiveDetector);
