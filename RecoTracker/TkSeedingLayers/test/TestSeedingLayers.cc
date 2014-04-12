#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <string>
#include <iostream>

struct TestSeedingLayers final : public edm::EDAnalyzer {


  explicit TestSeedingLayers(const edm::ParameterSet& conf) {

  }

  virtual ~TestSeedingLayers(){}
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) {
    edm::Handle<SeedingLayerSetsHits> layersH;

    std::string layerProd = "MixedLayerTriplets";
    e.getByLabel(layerProd,layersH);
    
    auto const & layers = *layersH;

    std::cout << layers.numberOfLayersInSet() << ' ' << layers.size() << std::endl;
    
    for (auto const & lset : layers) {
      std::cout << lset.size();
      for (auto const & la : lset) {
	std::cout << ": " << la.name() << ' ' << la.index() << ' ' << la.hits().size();
      }
      std::cout << std::endl;
    }
  }
  

    

};


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestSeedingLayers);
