#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CombinatorialRegionalSeedGeneratorFromPixel::init(const SiPixelRecHitCollection &coll , const edm::EventSetup& iSetup)
{
  pixelLayers->init(coll,iSetup);
  initPairGenerator(pixelLayers,iSetup);
}

CombinatorialRegionalSeedGeneratorFromPixel::CombinatorialRegionalSeedGeneratorFromPixel(edm::ParameterSet const& conf)
  : SeedGeneratorFromLayerPairs(conf)
{  
  edm::ParameterSet conf_ = pSet();



  pixelLayers = new PixelSeedLayerPairs();
}

void CombinatorialRegionalSeedGeneratorFromPixel::run(RectangularEtaPhiTrackingRegion& etaphiRegion, TrajectorySeedCollection &output, const edm::EventSetup& iSetup){

  region = etaphiRegion;
  seeds(output, iSetup,region);
}
