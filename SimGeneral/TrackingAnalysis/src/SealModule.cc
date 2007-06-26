#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"
DEFINE_ANOTHER_FWK_MODULE(TrackingTruthProducer);

#include "SimGeneral/TrackingAnalysis/interface/TrackingElectronProducer.h"
DEFINE_ANOTHER_FWK_MODULE(TrackingElectronProducer);
