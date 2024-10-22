#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "PixelTripletHLTGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory, PixelTripletHLTGenerator, "PixelTripletHLTGenerator");

#include "PixelTripletLargeTipGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory,
                  PixelTripletLargeTipGenerator,
                  "PixelTripletLargeTipGenerator");

#include "PixelTripletNoTipGenerator.h"
DEFINE_EDM_PLUGIN(HitTripletGeneratorFromPairAndLayersFactory,
                  PixelTripletNoTipGenerator,
                  "PixelTripletNoTipGenerator");

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "CombinedHitTripletGenerator.h"
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, CombinedHitTripletGenerator, "StandardHitTripletGenerator");

#include "RecoTracker/PixelSeeding/interface/HitTripletEDProducerT.h"
using PixelTripletHLTEDProducer = HitTripletEDProducerT<PixelTripletHLTGenerator>;
DEFINE_FWK_MODULE(PixelTripletHLTEDProducer);

using PixelTripletLargeTipEDProducer = HitTripletEDProducerT<PixelTripletLargeTipGenerator>;
DEFINE_FWK_MODULE(PixelTripletLargeTipEDProducer);
