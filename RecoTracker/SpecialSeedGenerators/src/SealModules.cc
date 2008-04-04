#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
#include "RecoTracker/SpecialSeedGenerators/interface/CtfSpecialSeedGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicSeedGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CRackSeedGenerator.h"

DEFINE_ANOTHER_FWK_MODULE(CtfSpecialSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(CosmicSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(CRackSeedGenerator);

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/GenericTripletGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/GenericPairGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/BeamHaloPairGenerator.h"

DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, GenericTripletGenerator, "GenericTripletGenerator");
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, GenericPairGenerator, "GenericPairGenerator");
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, BeamHaloPairGenerator, "BeamHaloPairGenerator");

