#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociatorBaseImpl.h"

namespace hgcal {
  TracksterToSimTracksterHitLCAssociatorBaseImpl::TracksterToSimTracksterHitLCAssociatorBaseImpl(){};
  TracksterToSimTracksterHitLCAssociatorBaseImpl::~TracksterToSimTracksterHitLCAssociatorBaseImpl(){};

  hgcal::association_t TracksterToSimTracksterHitLCAssociatorBaseImpl::makeConnections(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return hgcal::association_t();
  }

  hgcal::RecoToSimCollectionSimTracksters TracksterToSimTracksterHitLCAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return hgcal::RecoToSimCollectionSimTracksters();
  }

  hgcal::SimToRecoCollectionSimTracksters TracksterToSimTracksterHitLCAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return hgcal::SimToRecoCollectionSimTracksters();
  }

}  // namespace hgcal
