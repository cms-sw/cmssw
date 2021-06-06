// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociatorBaseImpl.h"

namespace hgcal {
  TracksterToSimClusterAssociatorBaseImpl::TracksterToSimClusterAssociatorBaseImpl(){};
  TracksterToSimClusterAssociatorBaseImpl::~TracksterToSimClusterAssociatorBaseImpl(){};

  hgcal::RecoToSimCollectionTracksters TracksterToSimClusterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH) const {
    return hgcal::RecoToSimCollectionTracksters();
  }

  hgcal::SimToRecoCollectionTracksters TracksterToSimClusterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH) const {
    return hgcal::SimToRecoCollectionTracksters();
  }

}  // namespace hgcal
