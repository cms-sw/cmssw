#ifndef SimDataFormats_Associations_LayerClusterToCaloParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_LayerClusterToCaloParticleAssociatorBaseImpl_h

/** \class LayerClusterToCaloParticleAssociatorBaseImpl
 *
 * Base class for LayerClusterToCaloParticleAssociators.  Methods take as input
 * the handle of LayerClusters and the CaloParticle collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Marco Rovere
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

namespace hgcal {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<CaloParticleCollection, reco::CaloClusterCollection, std::pair<float, float>>>
      SimToRecoCollection;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<reco::CaloClusterCollection, CaloParticleCollection, float>>
      RecoToSimCollection;

  class LayerClusterToCaloParticleAssociatorBaseImpl {
  public:
    /// Constructor
    LayerClusterToCaloParticleAssociatorBaseImpl();
    /// Destructor
    virtual ~LayerClusterToCaloParticleAssociatorBaseImpl();

    /// Associate a LayerCluster to CaloParticles
    virtual hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                          const edm::Handle<CaloParticleCollection> &cPCH) const;

    /// Associate a CaloParticle to LayerClusters
    virtual hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                          const edm::Handle<CaloParticleCollection> &cPCH) const;
  };
}  // namespace hgcal

#endif
