#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

typedef DreamSD DreamSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(DreamSensitiveDetector);
