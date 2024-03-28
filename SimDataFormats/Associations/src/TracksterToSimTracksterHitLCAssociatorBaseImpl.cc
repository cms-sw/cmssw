#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociatorBaseImpl.h"

namespace ticl {
  TracksterToSimTracksterHitLCAssociatorBaseImpl::TracksterToSimTracksterHitLCAssociatorBaseImpl(){};
  TracksterToSimTracksterHitLCAssociatorBaseImpl::~TracksterToSimTracksterHitLCAssociatorBaseImpl(){};

  ticl::association_t TracksterToSimTracksterHitLCAssociatorBaseImpl::makeConnections(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return ticl::association_t();
  }

  ticl::RecoToSimCollectionSimTracksters TracksterToSimTracksterHitLCAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return ticl::RecoToSimCollectionSimTracksters();
  }

  ticl::SimToRecoCollectionSimTracksters TracksterToSimTracksterHitLCAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return ticl::SimToRecoCollectionSimTracksters();
  }

}  // namespace ticl
