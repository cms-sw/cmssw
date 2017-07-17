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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <memory>

namespace muonAssociatorByHitsDiagnostics {
  class InputDumper;
}

class MuonAssociatorByHits {
  
 public:
  
  MuonAssociatorByHits (const edm::ParameterSet& conf, edm::ConsumesCollector && iC);   
  ~MuonAssociatorByHits();
  
  // Originally from TrackAssociatorBase from where this class used to inherit from
  reco::RecoToSimCollection associateRecoToSim(edm::Handle<edm::View<reco::Track> >& tCH,
						       edm::Handle<TrackingParticleCollection>& tPCH,
						       const edm::Event * event ,
                                                       const edm::EventSetup * setup ) const {
    edm::RefToBaseVector<reco::Track> tc;
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(tCH->refAt(j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateRecoToSim(tc,tpc,event,setup);
  }

  virtual reco::SimToRecoCollection associateSimToReco(edm::Handle<edm::View<reco::Track> >& tCH,
						       edm::Handle<TrackingParticleCollection>& tPCH,
						       const edm::Event * event ,
                                                       const edm::EventSetup * setup ) const {
    edm::RefToBaseVector<reco::Track> tc;
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(tCH->refAt(j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateSimToReco(tc,tpc,event,setup);
  }

 
  /* Associate SimTracks to RecoTracks By Hits */
  /// Association Reco To Sim with Collections
  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const;
  
  /// Association Sim To Reco with Collections
  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const;

 
 private:
  MuonAssociatorByHitsHelper helper_;
  edm::ParameterSet const conf_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;

  std::unique_ptr<muonAssociatorByHitsDiagnostics::InputDumper> diagnostics_;
};

#endif
