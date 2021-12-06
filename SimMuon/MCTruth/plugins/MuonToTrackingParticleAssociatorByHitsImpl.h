#ifndef SimMuon_MCTruth_MuonToTrackingParticleAssociatorByHitsImpl_h
#define SimMuon_MCTruth_MuonToTrackingParticleAssociatorByHitsImpl_h
// -*- C++ -*-
//
// Package:     SimMuon/MCTruth
// Class  :     MuonToTrackingParticleAssociatorByHitsImpl
//
/**\class MuonToTrackingParticleAssociatorByHitsImpl
 MuonToTrackingParticleAssociatorByHitsImpl.h
 "MuonToTrackingParticleAssociatorByHitsImpl.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:35:52 GMT
//

// system include files
#include <functional>

// user include files
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociatorBaseImpl.h"
#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"

// forward declarations
class TrackerMuonHitExtractor;

class MuonToTrackingParticleAssociatorByHitsImpl : public reco::MuonToTrackingParticleAssociatorBaseImpl {
public:
  using TrackHitsCollection = MuonAssociatorByHitsHelper::TrackHitsCollection;

  MuonToTrackingParticleAssociatorByHitsImpl(
      TrackerMuonHitExtractor const &iHitExtractor,
      TrackerHitAssociator::Config const &iTracker,
      CSCHitAssociator::Config const &iCSC,
      DTHitAssociator::Config const &iDT,
      RPCHitAssociator::Config const &iRPC,
      GEMHitAssociator::Config const &iGEM,
      edm::Event const &iEvent,
      edm::EventSetup const &iSetup,
      const TrackerTopology *iTopo,
      std::function<void(const TrackHitsCollection &, const TrackingParticleCollection &)>,
      MuonAssociatorByHitsHelper const *iHelper);

  MuonToTrackingParticleAssociatorByHitsImpl(const MuonToTrackingParticleAssociatorByHitsImpl &) =
      delete;  // stop default

  const MuonToTrackingParticleAssociatorByHitsImpl &operator=(const MuonToTrackingParticleAssociatorByHitsImpl &) =
      delete;  // stop default

  // ---------- const member functions ---------------------
  void associateMuons(reco::MuonToSimCollection &recoToSim,
                      reco::SimToMuonCollection &simToReco,
                      const edm::RefToBaseVector<reco::Muon> &muons,
                      reco::MuonTrackType type,
                      const edm::RefVector<TrackingParticleCollection> &tpColl) const override;

  void associateMuons(reco::MuonToSimCollection &recoToSim,
                      reco::SimToMuonCollection &simToReco,
                      const edm::Handle<edm::View<reco::Muon>> &muons,
                      reco::MuonTrackType type,
                      const edm::Handle<TrackingParticleCollection> &tpColl) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

private:
  // ---------- member data --------------------------------
  TrackerMuonHitExtractor const *m_hitExtractor;
  TrackerHitAssociator const m_tracker;
  CSCHitAssociator const m_csc;
  DTHitAssociator const m_dt;
  RPCHitAssociator const m_rpc;
  GEMHitAssociator const m_gem;
  MuonAssociatorByHitsHelper::Resources m_resources;
  MuonAssociatorByHitsHelper const *m_helper;
};

#endif
