#ifndef SimDataFormats_Associations_TracksterToSimClusterAssociatorBaseImpl_h
#define SimDataFormats_Associations_TracksterToSimClusterAssociatorBaseImpl_h

/** \class TracksterToSimClusterAssociatorBaseImpl
 *
 * Base class for TracksterToSimClusterAssociators.  Methods take as input
 * the handle of Tracksters and the SimCluster collections and return an
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
      edm::OneToManyWithQualityGeneric<SimClusterCollection, ticl::TracksterCollection, std::pair<float, float>>>
      SimToRecoCollectionTracksters;
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric<ticl::TracksterCollection, SimClusterCollection, float>>
      RecoToSimCollectionTracksters;

  class TracksterToSimClusterAssociatorBaseImpl {
  public:
    /// Constructor
    TracksterToSimClusterAssociatorBaseImpl();
    /// Destructor
    virtual ~TracksterToSimClusterAssociatorBaseImpl();

    /// Associate a Trackster to SimClusters
    virtual hgcal::RecoToSimCollectionTracksters associateRecoToSim(
        const edm::Handle<ticl::TracksterCollection> &tCH, const edm::Handle<reco::CaloClusterCollection> &lCCH, const edm::Handle<SimClusterCollection> &sCCH) const;

    /// Associate a SimCluster to Tracksters
    virtual hgcal::SimToRecoCollectionTracksters associateSimToReco(
        const edm::Handle<ticl::TracksterCollection> &tCH, const edm::Handle<reco::CaloClusterCollection> &lCCH, const edm::Handle<SimClusterCollection> &sCCH) const;
  };
}  // namespace hgcal

#endif
