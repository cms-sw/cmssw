#ifndef MCTruth_MuonAssociatorEDProducer_h
#define MCTruth_MuonAssociatorEDProducer_h

#include <memory>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"

class MuonAssociatorEDProducer : public edm::EDProducer {
public:
  explicit MuonAssociatorEDProducer(const edm::ParameterSet&);
  ~MuonAssociatorEDProducer();
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  edm::InputTag tracksTag;
  edm::InputTag tpTag;
  bool ignoreMissingTrackCollection;
  edm::ParameterSet parset_;
  MuonAssociatorByHits * associatorByHits;
};

#endif
