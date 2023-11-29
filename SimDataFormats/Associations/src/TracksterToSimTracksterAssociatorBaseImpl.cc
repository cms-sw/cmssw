// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociatorBaseImpl.h"

namespace ticl {
  TracksterToSimTracksterAssociatorBaseImpl::TracksterToSimTracksterAssociatorBaseImpl(){};
  TracksterToSimTracksterAssociatorBaseImpl::~TracksterToSimTracksterAssociatorBaseImpl(){};

  ticl::RecoToSimCollectionSimTracksters TracksterToSimTracksterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return ticl::RecoToSimCollectionSimTracksters();
  }

  ticl::SimToRecoCollectionSimTracksters TracksterToSimTracksterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const {
    return ticl::SimToRecoCollectionSimTracksters();
  }

}  // namespace ticl
