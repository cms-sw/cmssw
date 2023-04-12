#include "RecoTracker/PixelSeeding/interface/CosmicLayerTriplets.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/PixelSeeding/interface/CosmicHitTripletGenerator.h"

#include <vector>

using namespace std;

CosmicHitTripletGenerator::CosmicHitTripletGenerator(CosmicLayerTriplets& layers, const TrackerGeometry& trackGeom) {
  //  vector<LayerTriplets::LayerTriplet> layerTriplets = layers();
  vector<CosmicLayerTriplets::LayerPairAndLayers> layerTriplets = layers.layers();
  vector<CosmicLayerTriplets::LayerPairAndLayers>::const_iterator it;
  for (it = layerTriplets.begin(); it != layerTriplets.end(); it++) {
    vector<const LayerWithHits*>::const_iterator ilwh;
    for (ilwh = (*it).second.begin(); ilwh != (*it).second.end(); ilwh++) {
      //      const LayerWithHits* first=(*it).first.first;
      //       const LayerWithHits* second=(*it).first.second;
      //       const LayerWithHits* third=(*ilwh);
      //      add( (*it).first.first, (*it).first.second, (*it).second,iSetup);
      add((*it).first.first, (*it).first.second, (*ilwh), trackGeom);
    }
  }
}

CosmicHitTripletGenerator::~CosmicHitTripletGenerator() {}

void CosmicHitTripletGenerator::add(const LayerWithHits* inner,
                                    const LayerWithHits* middle,
                                    const LayerWithHits* outer,
                                    const TrackerGeometry& trackGeom) {
  theGenerators.push_back(std::make_unique<CosmicHitTripletGeneratorFromLayerTriplet>(inner, middle, outer, trackGeom));
}

void CosmicHitTripletGenerator::hitTriplets(const TrackingRegion& region, OrderedHitTriplets& pairs) {
  Container::const_iterator i;
  for (i = theGenerators.begin(); i != theGenerators.end(); i++) {
    (**i).hitTriplets(region, pairs);
  }
}
