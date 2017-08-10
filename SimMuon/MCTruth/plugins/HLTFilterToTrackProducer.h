//
// modified & integrated by C. Battilana (INFN BO)
// from code by G. Abbiendi: SimMuon/MCTruth/plugins/MuonTrackSelector.cc
//
#ifndef MCTruth_HLTFilterToTrackProducer_h
#define MCTruth_HLTFilterToTrackProducer_h

#include <memory>
#include <string>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

class HLTFilterToTrackProducer : public edm::stream::EDProducer<> 
{

 public:

  explicit HLTFilterToTrackProducer(const edm::ParameterSet&);
  virtual ~HLTFilterToTrackProducer();

 private:
 
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<trigger::TriggerEvent> m_trigEvToken;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_filterToken;
  std::string m_filterName;

};

#endif
