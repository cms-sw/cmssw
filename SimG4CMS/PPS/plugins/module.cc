#include "SimG4CMS/PPS/interface/TotemRPSD.h"
#include "SimG4CMS/PPS/interface/PPSPixelSD.h"
#include "SimG4CMS/PPS/interface/PPSDiamondSD.h"

#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

typedef PPSPixelSD CTPPSSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CTPPSSensitiveDetector);

typedef TotemRPSD RomanPotSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(RomanPotSensitiveDetector);

typedef PPSDiamondSD CTPPSDiamondSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CTPPSDiamondSensitiveDetector);
