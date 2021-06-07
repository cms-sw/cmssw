#ifndef SimDataFormats_Associations_LayerClusterToSimTracksterAssociatorBaseImpl_h
#define SimDataFormats_Associations_LayerClusterToSimTracksterAssociatorBaseImpl_h

/** \class LayerClusterToSimTracksterAssociatorBaseImpl
 *
 * Base class for LayerClusterToSimTracksterAssociators.  Methods take as input
 * the handle of LayerClusters and the SimTrackster collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Marco Rovere
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

namespace hgcal {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<CaloParticleCollection, reco::CaloClusterCollection, std::pair<float, float>>>
      SimToRecoCollection;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<reco::CaloClusterCollection, ticl::TracksterCollection, float>>
      RecoToSimCollection;

  class LayerClusterToSimTracksterAssociatorBaseImpl {
  public:
    /// Constructor
    LayerClusterToSimTracksterAssociatorBaseImpl();
    /// Destructor
    virtual ~LayerClusterToSimTracksterAssociatorBaseImpl();

    /// Associate a LayerCluster to SimTracksters
    virtual hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                          const edm::Handle<ticl::TracksterCollection> &sTCH) const;

    /// Associate a SimTrackster to LayerClusters
    virtual hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                          const edm::Handle<CaloParticleCollection> &cPCH) const;
  };
}  // namespace hgcal

#endif
