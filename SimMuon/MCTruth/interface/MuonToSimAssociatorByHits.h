#ifndef MuonToSimAssociatorByHits_h
#define MuonToSimAssociatorByHits_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimMuon/MCTruth/interface/MuonToSimAssociatorBase.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"

class MuonToSimAssociatorByHits : public MuonToSimAssociatorBase {
  
 public:
  
  MuonToSimAssociatorByHits (const edm::ParameterSet& conf, edm::ConsumesCollector && iC);   
  MuonToSimAssociatorByHits (const edm::ParameterSet& conf);   
  ~MuonToSimAssociatorByHits();



  void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                      const edm::RefToBaseVector<reco::Muon> &, MuonTrackType ,
                      const edm::RefVector<TrackingParticleCollection>&,
                      const edm::Event * event = 0, const edm::EventSetup * setup = 0) const override ; 

  void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                      const edm::Handle<edm::View<reco::Muon> > &, MuonTrackType , 
                      const edm::Handle<TrackingParticleCollection>&,
                      const edm::Event * event = 0, const edm::EventSetup * setup = 0) const override;

 private:

  MuonAssociatorByHitsHelper helper_;
  edm::ParameterSet const conf_;

};

#endif
