#ifndef SimDataFormats_Associations_TracksterToSimTracksterAssociatorBaseImpl_h
#define SimDataFormats_Associations_TracksterToSimTracksterAssociatorBaseImpl_h

/** \class TracksterToSimTracksterAssociatorBaseImpl
 *
 * Base class for TracksterToSimTracksterAssociator. Methods take as input
 * the handles of Tracksters, LayerClusters and SimTracksters collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Leonardo Cristella
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace hgcal {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<ticl::TracksterCollection, ticl::TracksterCollection, std::pair<float, float>>>
      SimToRecoCollectionSimTracksters;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<ticl::TracksterCollection, ticl::TracksterCollection, std::pair<float, float>>>
      RecoToSimCollectionSimTracksters;

  class TracksterToSimTracksterAssociatorBaseImpl {
  public:
    /// Constructor
    TracksterToSimTracksterAssociatorBaseImpl();
    /// Destructor
    virtual ~TracksterToSimTracksterAssociatorBaseImpl();

    /// Associate a Trackster to SimClusters
    virtual hgcal::RecoToSimCollectionSimTracksters associateRecoToSim(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const;

    /// Associate a SimCluster to Tracksters
    virtual hgcal::SimToRecoCollectionSimTracksters associateSimToReco(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const;
  };
}  // namespace hgcal

#endif
