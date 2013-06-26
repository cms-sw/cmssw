#include "SimG4CMS/ShowerLibraryProducer/interface/HFWedgeSD.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/HFChamberSD.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/FiberSD.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardAnalysis.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/CastorShowerLibraryMaker.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"


typedef HFWedgeSD HFWedgeSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HFWedgeSensitiveDetector);
typedef HFChamberSD HFChamberSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HFChamberSensitiveDetector);
typedef FiberSD FiberSensitiveDetector;
DEFINE_SENSITIVEDETECTOR(FiberSensitiveDetector);
DEFINE_SIMWATCHER (HcalForwardAnalysis);
DEFINE_SIMWATCHER (CastorShowerLibraryMaker);
