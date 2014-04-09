#include "SimG4CMS/Forward/interface/CastorSD.h"
#include "SimG4CMS/Forward/interface/TotemSD.h"
#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "SimG4CMS/Forward/interface/TotemTestGem.h"
#include "SimG4CMS/Forward/interface/CastorTestAnalysis.h"
#include "SimG4CMS/Forward/interface/ZdcTestAnalysis.h"
#include "SimG4CMS/Forward/interface/DoCastorAnalysis.h"
#include "SimG4CMS/Forward/interface/PltSD.h"

#include "SimG4CMS/Forward/interface/BscTest.h"
#include "SimG4CMS/Forward/interface/BscSD.h"
#include "SimG4CMS/Forward/interface/Bcm1fSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
  

typedef CastorSD CastorSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(CastorSensitiveDetector);
typedef TotemSD TotemSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(TotemSensitiveDetector);
typedef ZdcSD ZdcSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(ZdcSensitiveDetector);
typedef BscSD BSCSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(BSCSensitiveDetector);
typedef PltSD PLTSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(PLTSensitiveDetector);
typedef Bcm1fSD BCM1FSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(BCM1FSensitiveDetector);
DEFINE_SIMWATCHER (CastorTestAnalysis);
DEFINE_SIMWATCHER (ZdcTestAnalysis);
DEFINE_SIMWATCHER (DoCastorAnalysis);
DEFINE_SIMWATCHER (TotemTestGem);
DEFINE_SIMWATCHER (BscTest);
