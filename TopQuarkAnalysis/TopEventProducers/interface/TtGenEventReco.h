#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TtGenEventReco : public edm::EDProducer {

 public:

  explicit TtGenEventReco(const edm::ParameterSet&);
  ~TtGenEventReco();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::InputTag src_;
  edm::InputTag init_;
};
