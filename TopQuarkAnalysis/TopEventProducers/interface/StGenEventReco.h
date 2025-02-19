#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class StGenEventReco : public edm::EDProducer {

 public:

  explicit StGenEventReco(const edm::ParameterSet&);
  ~StGenEventReco();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  edm::InputTag src_;
  edm::InputTag init_; 
};
