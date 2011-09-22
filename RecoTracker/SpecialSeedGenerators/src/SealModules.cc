#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CtfSpecialSeedGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicSeedGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CRackSeedGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/SimpleCosmicBONSeeder.h"

DEFINE_FWK_MODULE(CtfSpecialSeedGenerator);
DEFINE_FWK_MODULE(CosmicSeedGenerator);
DEFINE_FWK_MODULE(CRackSeedGenerator);
DEFINE_FWK_MODULE(SimpleCosmicBONSeeder);

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/GenericTripletGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/GenericPairGenerator.h"
#include "RecoTracker/SpecialSeedGenerators/interface/BeamHaloPairGenerator.h"

DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, GenericTripletGenerator, "GenericTripletGenerator");
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, GenericPairGenerator, "GenericPairGenerator");
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, BeamHaloPairGenerator, "BeamHaloPairGenerator");

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h" 	 
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h" 	
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicRegionalSeedGenerator.h" 

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, CosmicRegionalSeedGenerator, "CosmicRegionalSeedGenerator"); 

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicSeedCreator.h"

DEFINE_EDM_PLUGIN(SeedCreatorFactory, CosmicSeedCreator, "CosmicSeedCreator");
