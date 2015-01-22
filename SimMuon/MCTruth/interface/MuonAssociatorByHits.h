#ifndef MuonAssociatorByHits_h
#define MuonAssociatorByHits_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"

#include <memory>

namespace muonAssociatorByHitsDiagnostics {
  class InputDumper;
}

class MuonAssociatorByHits : public TrackAssociatorBase {
  
 public:
  
  MuonAssociatorByHits (const edm::ParameterSet& conf, edm::ConsumesCollector && iC);   
  MuonAssociatorByHits (const edm::ParameterSet& conf);   
  ~MuonAssociatorByHits();
  
  // Get base methods from base class
  using TrackAssociatorBase::associateRecoToSim;
  using TrackAssociatorBase::associateSimToReco;
 
  /* Associate SimTracks to RecoTracks By Hits */
  /// Association Reco To Sim with Collections
  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const override;
  
  /// Association Sim To Reco with Collections
  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const override;

 
 private:
  MuonAssociatorByHitsHelper helper_;
  edm::ParameterSet const conf_;

  std::unique_ptr<muonAssociatorByHitsDiagnostics::InputDumper> diagnostics_;
};

#endif
