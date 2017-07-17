#ifndef SimDataFormats_Associations_TrackToTrackingParticleAssociator_h
#define SimDataFormats_Associations_TrackToTrackingParticleAssociator_h
// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     TrackToTrackingParticleAssociator
// 
/**\class TrackToTrackingParticleAssociator TrackToTrackingParticleAssociator.h "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

 Description: Interface for accessing a Track to TrackingParticle associator

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 30 Dec 2014 20:47:00 GMT
//

// system include files
#include <memory>

// user include files
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"

// forward declarations

namespace reco {

  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <TrackingParticleCollection, edm::View<TrajectorySeed>, double> >
    SimToRecoCollectionSeed;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <edm::View<TrajectorySeed>, TrackingParticleCollection, double> >
    RecoToSimCollectionSeed;  
  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <TrackingParticleCollection, TrackCandidateCollection, double> >
    SimToRecoCollectionTCandidate;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <TrajectorySeedCollection, TrackCandidateCollection, double> >
    RecoToSimCollectionTCandidate;  

  class TrackToTrackingParticleAssociator
  {
    
  public:

#ifndef __GCCXML__
    TrackToTrackingParticleAssociator( std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl>);
#endif
    TrackToTrackingParticleAssociator();
    ~TrackToTrackingParticleAssociator();
    
    // ---------- const member functions ---------------------
    /// compare reco to sim the handle of reco::Track and TrackingParticle collections
    reco::RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                 const edm::Handle<TrackingParticleCollection>& tPCH ) const {
      return m_impl->associateRecoToSim(tCH,tPCH);
    }
    
    /// compare reco to sim the handle of reco::Track and TrackingParticle collections
    reco::SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
                                                 const edm::Handle<TrackingParticleCollection>& tPCH ) const {
      return m_impl->associateSimToReco(tCH,tPCH);
    }

    /// Association Reco To Sim with Collections
    reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track> & tc,
                                                 const edm::RefVector<TrackingParticleCollection>& tpc) const {
      return m_impl->associateRecoToSim(tc,tpc);
    }

    /// Association Sim To Reco with Collections
    reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track> & tc,
                                                 const edm::RefVector<TrackingParticleCollection>& tpc ) const {
      return m_impl->associateSimToReco(tc,tpc);
    }

    //TrajectorySeed
    reco::RecoToSimCollectionSeed associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed> >& ts, 
                                                     const edm::Handle<TrackingParticleCollection>& tpc) const {
      return m_impl->associateRecoToSim(ts, tpc);
    }

    reco::SimToRecoCollectionSeed associateSimToReco(const edm::Handle<edm::View<TrajectorySeed> >& ts, 
                                                     const edm::Handle<TrackingParticleCollection>& tpc) const {
      return m_impl->associateSimToReco(ts,tpc);
    }

    //TrackCandidate
    reco::RecoToSimCollectionTCandidate associateRecoToSim(const edm::Handle<TrackCandidateCollection>& tc, 
                                                           const edm::Handle<TrackingParticleCollection>& tpc) const {
      return m_impl->associateRecoToSim(tc,tpc);
    }
  
    reco::SimToRecoCollectionTCandidate associateSimToReco(const edm::Handle<TrackCandidateCollection>& tc, 
                                                           const edm::Handle<TrackingParticleCollection>& tpc) const {
      return m_impl->associateSimToReco(tc,tpc);
    }

    void swap(TrackToTrackingParticleAssociator& iOther) {
      std::swap(m_impl, iOther.m_impl);
    }
  private:
    TrackToTrackingParticleAssociator(const TrackToTrackingParticleAssociator&); // stop default
    
    const TrackToTrackingParticleAssociator& operator=(const TrackToTrackingParticleAssociator&); // stop default
    
    // ---------- member data --------------------------------
    TrackToTrackingParticleAssociatorBaseImpl* m_impl;
  };
}

#endif
