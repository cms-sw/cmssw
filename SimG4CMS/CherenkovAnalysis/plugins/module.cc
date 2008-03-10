#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE ();
typedef DreamSD DreamSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(DreamSensitiveDetector);
