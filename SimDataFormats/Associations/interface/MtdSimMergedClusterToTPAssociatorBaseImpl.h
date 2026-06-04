#ifndef SimDataFormats_Associations_MtdSimMergedClusterToTPAssociatorBaseImpl_h
#define SimDataFormats_Associations_MtdSimMergedClusterToTPAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  typedef edm::AssociationMap<edm::OneToMany<MtdSimMergedClusterCollection, TrackingParticleCollection> >
      MergedSimToTPCollectionMtd;
  typedef edm::AssociationMap<edm::OneToMany<TrackingParticleCollection, MtdSimMergedClusterCollection> >
      TPToMergedSimCollectionMtd;

  class MtdSimMergedClusterToTPAssociatorBaseImpl {
  public:
    /// Constructor
    MtdSimMergedClusterToTPAssociatorBaseImpl();
    /// Destructor
    virtual ~MtdSimMergedClusterToTPAssociatorBaseImpl();

    /// Associate a MtdSimMergedCluster to TrackingParticle
    virtual MergedSimToTPCollectionMtd associateSimToTP(
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const;

    /// Associate a TrackingParticle to MtdSimMergedCluster
    virtual TPToMergedSimCollectionMtd associateTPToSim(
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const;
  };
}  // namespace reco

#endif
