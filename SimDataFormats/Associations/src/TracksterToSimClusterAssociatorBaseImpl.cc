// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociatorBaseImpl.h"

namespace ticl {
  TracksterToSimClusterAssociatorBaseImpl::TracksterToSimClusterAssociatorBaseImpl(){};
  TracksterToSimClusterAssociatorBaseImpl::~TracksterToSimClusterAssociatorBaseImpl(){};

  ticl::RecoToSimCollectionTracksters TracksterToSimClusterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH) const {
    return ticl::RecoToSimCollectionTracksters();
  }

  ticl::SimToRecoCollectionTracksters TracksterToSimClusterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH) const {
    return ticl::SimToRecoCollectionTracksters();
  }

}  // namespace ticl
