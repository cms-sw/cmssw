#ifndef Validation_EventGenerator_WeightManager 
#define Validation_EventGenerator_WeightManager

// Utility class, that computes the event weight, 
// either returning the weight as stored in the HepMCCollection
// or returning the product of the weights stored in 
// a vector of GenEventInfoProducts

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <vector>

namespace edm{
  class ParameterSet;
  class Event;
}

class WeightManager {

 public:

  WeightManager( const edm::ParameterSet&,edm::ConsumesCollector iC);
  ~WeightManager(){};

  double weight(const edm::Event&);
  
 private:
  bool _useHepMC;
  std::vector<edm::InputTag> _genEventInfos;
  edm::InputTag _hepmcCollection;

  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
  std::vector<edm::EDGetTokenT<std::vector<edm::InputTag> > > genEventInfosTokens_;

};

#endif

