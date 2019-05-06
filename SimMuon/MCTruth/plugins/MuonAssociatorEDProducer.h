#ifndef MCTruth_MuonAssociatorEDProducer_h
#define MCTruth_MuonAssociatorEDProducer_h

#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include <memory>

class MuonAssociatorEDProducer : public edm::stream::EDProducer<> {
public:
  explicit MuonAssociatorEDProducer(const edm::ParameterSet &);
  ~MuonAssociatorEDProducer() override;

private:
  virtual void beginJob();
  void produce(edm::Event &, const edm::EventSetup &) override;
  virtual void endJob();

  edm::InputTag tracksTag;
  edm::InputTag tpTag;
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;

  bool ignoreMissingTrackCollection;
  edm::ParameterSet parset_;
  MuonAssociatorByHits *associatorByHits;
};

#endif
