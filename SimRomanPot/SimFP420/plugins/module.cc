#include "SimRomanPot/SimFP420/interface/DigFP420Test.h"
#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
  
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER (DigFP420Test);
DEFINE_SIMWATCHER (DigitizerFP420); //=
//DEFINE_FWK_MODULE(DigFP420Test);
//DEFINE_FWK_MODULE(DigitizerFP420);
