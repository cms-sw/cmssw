#ifndef SimDataFormats_Associations_LayerClusterToSimTracksterAssociatorBaseImpl_h
#define SimDataFormats_Associations_LayerClusterToSimTracksterAssociatorBaseImpl_h

/** \class LayerClusterToSimTracksterAssociatorBaseImpl
 *
 * Base class for LayerClusterToSimTracksterAssociators. Methods take as input
 * the handle of LayerClusters and the SimTrackster collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Leonardo Cristella
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "LayerClusterToCaloParticleAssociatorBaseImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "LayerClusterToSimClusterAssociatorBaseImpl.h"

namespace ticl {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<ticl::TracksterCollection, reco::CaloClusterCollection, std::pair<float, float>>>
      SimTracksterToRecoCollection;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<reco::CaloClusterCollection, ticl::TracksterCollection, float>>
      RecoToSimTracksterCollection;

  class LayerClusterToSimTracksterAssociatorBaseImpl {
  public:
    /// Constructor
    LayerClusterToSimTracksterAssociatorBaseImpl();
    /// Destructor
    virtual ~LayerClusterToSimTracksterAssociatorBaseImpl();

    /// Associate a LayerCluster to SimTracksters
    virtual ticl::RecoToSimTracksterCollection associateRecoToSim(
        const edm::Handle<reco::CaloClusterCollection> &cCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const ticl::RecoToSimCollection &lCToCPs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const ticl::RecoToSimCollectionWithSimClusters &lCToSCs) const;

    /// Associate a SimTrackster to LayerClusters
    virtual ticl::SimTracksterToRecoCollection associateSimToReco(
        const edm::Handle<reco::CaloClusterCollection> &cCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const ticl::SimToRecoCollection &cPToLCs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const ticl::SimToRecoCollectionWithSimClusters &sCToLCs) const;
  };
}  // namespace ticl

#endif
