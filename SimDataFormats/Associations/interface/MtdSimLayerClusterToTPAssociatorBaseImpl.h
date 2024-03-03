#ifndef SimDataFormats_Associations_MtdSimLayerClusterToTPAssociatorBaseImpl_h
#define SimDataFormats_Associations_MtdSimLayerClusterToTPAssociatorBaseImpl_h

/** \class MtdSimLayerClusterToTPAssociatorBaseImpl
 *
 * Base class for MtdSimLayerClusterToTPAssociator. Methods take as input
 * the handles of MtdSimLayerCluster and TrackingParticle collections and return an
 * AssociationMap (oneToMany)
 *
 *  \author M. Malberti
 */

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  typedef edm::AssociationMap<edm::OneToMany<MtdSimLayerClusterCollection, TrackingParticleCollection> >
      SimToTPCollectionMtd;
  typedef edm::AssociationMap<edm::OneToMany<TrackingParticleCollection, MtdSimLayerClusterCollection> >
      TPToSimCollectionMtd;

  class MtdSimLayerClusterToTPAssociatorBaseImpl {
  public:
    /// Constructor
    MtdSimLayerClusterToTPAssociatorBaseImpl();
    /// Destructor
    virtual ~MtdSimLayerClusterToTPAssociatorBaseImpl();

    /// Associate a MtdSimLayerCluster to TrackingParticle
    virtual SimToTPCollectionMtd associateSimToTP(
        const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const;

    /// Associate a TrackingParticle to MtdSimLayerCluster
    virtual TPToSimCollectionMtd associateTPToSim(
        const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const;
  };
}  // namespace reco

#endif
