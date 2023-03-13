#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// Producers
//#include "PixelTrackProducerWithZPos.h"
//DEFINE_FWK_MODULE(PixelTrackProducerWithZPos);

#include "PixelVertexProducerMedian.h"
DEFINE_FWK_MODULE(PixelVertexProducerMedian);

#include "PixelVertexProducerClusters.h"
DEFINE_FWK_MODULE(PixelVertexProducerClusters);

#include "TrackListCombiner.h"
DEFINE_FWK_MODULE(TrackListCombiner);

// Generator
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/PixelLowPtUtilities/interface/PixelTripletLowPtGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory,
                  PixelTripletLowPtGenerator,
                  "PixelTripletLowPtGenerator");

// Seed
//#include "RecoTracker/PixelLowPtUtilities/interface/SeedProducer.h"
//DEFINE_FWK_MODULE(SeedProducer);

// TrajectoryFilter
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"
DEFINE_EDM_VALIDATED_PLUGIN(TrajectoryFilterFactory, ClusterShapeTrajectoryFilter, "ClusterShapeTrajectoryFilter");

// the seed comparitor to remove seeds on incompatible angle/cluster compatibility
#include "RecoTracker/PixelLowPtUtilities/interface/LowPtClusterShapeSeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
DEFINE_EDM_PLUGIN(SeedComparitorFactory, LowPtClusterShapeSeedComparitor, "LowPtClusterShapeSeedComparitor");

#include "RecoTracker/PixelLowPtUtilities/interface/StripSubClusterShapeTrajectoryFilter.h"
DEFINE_EDM_VALIDATED_PLUGIN(TrajectoryFilterFactory,
                            StripSubClusterShapeTrajectoryFilter,
                            "StripSubClusterShapeTrajectoryFilter");
DEFINE_EDM_PLUGIN(SeedComparitorFactory, StripSubClusterShapeSeedFilter, "StripSubClusterShapeSeedFilter");
