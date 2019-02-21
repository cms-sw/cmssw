#include "SimG4CMS/PPS/interface/Totem_RP_SD.h"
#include "SimG4CMS/PPS/interface/PPSPixelSD.h"
#include "SimG4CMS/PPS/interface/CTPPS_Diamond_SD.h"

#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
  
typedef PPSPixelSD CTPPSSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CTPPSSensitiveDetector);

typedef Totem_RP_SD RomanPotSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(RomanPotSensitiveDetector);

typedef CTPPS_Diamond_SD CTPPSDiamondSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CTPPSDiamondSensitiveDetector);
