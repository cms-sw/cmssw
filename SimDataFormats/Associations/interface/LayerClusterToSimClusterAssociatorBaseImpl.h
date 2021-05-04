#ifndef SimDataFormats_Associations_LayerClusterToSimClusterAssociatorBaseImpl_h
#define SimDataFormats_Associations_LayerClusterToSimClusterAssociatorBaseImpl_h

/** \class LayerClusterToSimClusterAssociatorBaseImpl
 *
 * Base class for LayerClusterToSimClusterAssociators.  Methods take as input
 * the handle of LayerClusters and the SimCluster collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Marco Rovere
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace hgcal {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<SimClusterCollection, reco::CaloClusterCollection, std::pair<float, float>>>
      SimToRecoCollectionWithSimClusters;
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric<reco::CaloClusterCollection, SimClusterCollection, float>>
      RecoToSimCollectionWithSimClusters;

  class LayerClusterToSimClusterAssociatorBaseImpl {
  public:
    /// Constructor
    LayerClusterToSimClusterAssociatorBaseImpl();
    /// Destructor
    virtual ~LayerClusterToSimClusterAssociatorBaseImpl();

    /// Associate a LayerCluster to SimClusters
    virtual hgcal::RecoToSimCollectionWithSimClusters associateRecoToSim(
        const edm::Handle<reco::CaloClusterCollection> &cCH, const edm::Handle<SimClusterCollection> &sCCH) const;

    /// Associate a SimCluster to LayerClusters
    virtual hgcal::SimToRecoCollectionWithSimClusters associateSimToReco(
        const edm::Handle<reco::CaloClusterCollection> &cCH, const edm::Handle<SimClusterCollection> &sCCH) const;
  };
}  // namespace hgcal

#endif
