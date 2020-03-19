#ifndef MuonToSimAssociatorByHits_h
#define MuonToSimAssociatorByHits_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/MCTruth/interface/MuonToSimAssociatorBase.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class MuonToSimAssociatorByHits : public MuonToSimAssociatorBase {
public:
  MuonToSimAssociatorByHits(const edm::ParameterSet &conf, edm::ConsumesCollector &&iC);
  ~MuonToSimAssociatorByHits() override;

  void associateMuons(MuonToSimCollection &recoToSim,
                      SimToMuonCollection &simToReco,
                      const edm::RefToBaseVector<reco::Muon> &,
                      MuonTrackType,
                      const edm::RefVector<TrackingParticleCollection> &,
                      const edm::Event *event = nullptr,
                      const edm::EventSetup *setup = nullptr) const override;

  void associateMuons(MuonToSimCollection &recoToSim,
                      SimToMuonCollection &simToReco,
                      const edm::Handle<edm::View<reco::Muon>> &,
                      MuonTrackType,
                      const edm::Handle<TrackingParticleCollection> &,
                      const edm::Event *event = nullptr,
                      const edm::EventSetup *setup = nullptr) const override;

private:
  MuonAssociatorByHitsHelper helper_;
  edm::ParameterSet const conf_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
};

#endif
