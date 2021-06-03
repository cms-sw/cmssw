// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociatorBaseImpl.h"

namespace hgcal {
  TracksterToSimTracksterAssociatorBaseImpl::TracksterToSimTracksterAssociatorBaseImpl(){};
  TracksterToSimTracksterAssociatorBaseImpl::~TracksterToSimTracksterAssociatorBaseImpl(){};

  hgcal::RecoToSimCollectionSimTracksters TracksterToSimTracksterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return hgcal::RecoToSimCollectionSimTracksters();
  }

  hgcal::SimToRecoCollectionSimTracksters TracksterToSimTracksterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return hgcal::SimToRecoCollectionSimTracksters();
  }

}  // namespace hgcal
