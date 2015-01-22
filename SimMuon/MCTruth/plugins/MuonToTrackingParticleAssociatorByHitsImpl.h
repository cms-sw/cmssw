#ifndef SimMuon_MCTruth_MuonToTrackingParticleAssociatorByHitsImpl_h
#define SimMuon_MCTruth_MuonToTrackingParticleAssociatorByHitsImpl_h
// -*- C++ -*-
//
// Package:     SimMuon/MCTruth
// Class  :     MuonToTrackingParticleAssociatorByHitsImpl
// 
/**\class MuonToTrackingParticleAssociatorByHitsImpl MuonToTrackingParticleAssociatorByHitsImpl.h "MuonToTrackingParticleAssociatorByHitsImpl.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:35:52 GMT
//

// system include files

// user include files
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociatorBaseImpl.h"
#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"

// forward declarations
class TrackerMuonHitExtractor;

class MuonToTrackingParticleAssociatorByHitsImpl: public reco::MuonToTrackingParticleAssociatorBaseImpl
{

   public:
   MuonToTrackingParticleAssociatorByHitsImpl(TrackerMuonHitExtractor const& iHitExtractor,
                                              MuonAssociatorByHitsHelper::Resources const& iResources,
                                              MuonAssociatorByHitsHelper const* iHelper);

   // ---------- const member functions ---------------------
   void associateMuons(reco::MuonToSimCollection & recoToSim, reco::SimToMuonCollection & simToReco,
                       const edm::RefToBaseVector<reco::Muon> & muons, reco::MuonTrackType type,
                       const edm::RefVector<TrackingParticleCollection>& tpColl) const override;
   
   void associateMuons(reco::MuonToSimCollection & recoToSim, reco::SimToMuonCollection & simToReco,
                       const edm::Handle<edm::View<reco::Muon> > & muons, reco::MuonTrackType type, 
                       const edm::Handle<TrackingParticleCollection>& tpColl) const override;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
 private:
   MuonToTrackingParticleAssociatorByHitsImpl(const MuonToTrackingParticleAssociatorByHitsImpl&); // stop default
   
   const MuonToTrackingParticleAssociatorByHitsImpl& operator=(const MuonToTrackingParticleAssociatorByHitsImpl&); // stop default
   
      // ---------- member data --------------------------------
   TrackerMuonHitExtractor const* m_hitExtractor;
   MuonAssociatorByHitsHelper::Resources m_resources;
   MuonAssociatorByHitsHelper const* m_helper;
};


#endif
