#ifndef SimDataFormats_Associations_MultiClusterToCaloParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_MultiClusterToCaloParticleAssociatorBaseImpl_h

/** \class MultiClusterToCaloParticleAssociatorBaseImpl
 *
 * Base class for MultiClusterToCaloParticleAssociators. Methods take as input
 * the handle of MultiClusters and the CaloParticle collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Leonardo Cristella
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

namespace hgcal {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<CaloParticleCollection, reco::HGCalMultiClusterCollection, std::pair<float, float>>>
      SimToRecoCollectionWithMultiClusters;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<reco::HGCalMultiClusterCollection, CaloParticleCollection, float>>
      RecoToSimCollectionWithMultiClusters;

  class MultiClusterToCaloParticleAssociatorBaseImpl {
  public:
    /// Constructor
    MultiClusterToCaloParticleAssociatorBaseImpl();
    /// Destructor
    virtual ~MultiClusterToCaloParticleAssociatorBaseImpl();

    /// Associate a MultiCluster to CaloParticles
    virtual hgcal::RecoToSimCollectionWithMultiClusters associateRecoToSim(
        const edm::Handle<reco::HGCalMultiClusterCollection> &cCH,
        const edm::Handle<CaloParticleCollection> &cPCH) const;

    /// Associate a CaloParticle to MultiClusters
    virtual hgcal::SimToRecoCollectionWithMultiClusters associateSimToReco(
        const edm::Handle<reco::HGCalMultiClusterCollection> &cCH,
        const edm::Handle<CaloParticleCollection> &cPCH) const;
  };
}  // namespace hgcal

#endif
