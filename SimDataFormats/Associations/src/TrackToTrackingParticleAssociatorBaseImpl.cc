// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     TrackToTrackingParticleAssociatorBaseImpl
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 30 Dec 2014 21:35:35 GMT
//

// system include files

// user include files
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
reco::TrackToTrackingParticleAssociatorBaseImpl::TrackToTrackingParticleAssociatorBaseImpl()
{
}

reco::TrackToTrackingParticleAssociatorBaseImpl::~TrackToTrackingParticleAssociatorBaseImpl()
{
}

//
// const member functions
//
reco::RecoToSimCollection 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateRecoToSim(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                                    const edm::Handle<TrackingParticleCollection>& tPCH) const 
{
  edm::RefToBaseVector<reco::Track> tc;
  for (unsigned int j=0; j<tCH->size();j++)
    tc.push_back(tCH->refAt(j));
  
  edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
  for (unsigned int j=0; j<tPCH->size();j++)
    tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
  
  return associateRecoToSim(tc,tpc);
}
    
/// compare reco to sim the handle of reco::Track and TrackingParticle collections
reco::SimToRecoCollection 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateSimToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                                    const edm::Handle<TrackingParticleCollection>& tPCH) const 
{
  edm::RefToBaseVector<reco::Track> tc;
  for (unsigned int j=0; j<tCH->size();j++)
    tc.push_back(tCH->refAt(j));
  
  edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
  for (unsigned int j=0; j<tPCH->size();j++)
    tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
  
  return associateSimToReco(tc,tpc);
}  
        
//TrajectorySeed
reco::RecoToSimCollectionSeed 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed> >&, 
                                                                    const edm::Handle<TrackingParticleCollection>& ) const
{
  reco::RecoToSimCollectionSeed empty;
  return empty;
}
    
reco::SimToRecoCollectionSeed 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateSimToReco(const edm::Handle<edm::View<TrajectorySeed> >&, 
                                                                    const edm::Handle<TrackingParticleCollection>&) const
{
  reco::SimToRecoCollectionSeed empty;
  return empty;
}
    
    //TrackCandidate
reco::RecoToSimCollectionTCandidate 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateRecoToSim(const edm::Handle<TrackCandidateCollection>&, 
                                                                    const edm::Handle<TrackingParticleCollection>&) const
{
  reco::RecoToSimCollectionTCandidate empty;
  return empty;
}
    
reco::SimToRecoCollectionTCandidate 
reco::TrackToTrackingParticleAssociatorBaseImpl::associateSimToReco(const edm::Handle<TrackCandidateCollection>&, 
                                                                    const edm::Handle<TrackingParticleCollection>&) const
{
  reco::SimToRecoCollectionTCandidate empty;
  return empty;
}
