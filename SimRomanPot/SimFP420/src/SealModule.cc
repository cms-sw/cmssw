//#include "PluginManager/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
////using cms::DigitizerFP420;
//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(DigitizerFP420);



#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
#include "SimRomanPot/SimFP420/interface/DigFP420Test.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
//#include "FWCore/PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
//typedef FP420SD FP420SensitiveDetector;
//DEFINE_SENSITIVEDETECTOR(FP420SensitiveDetector);
DEFINE_SIMWATCHER (DigitizerFP420); //=
DEFINE_SIMWATCHER (DigFP420Test);


