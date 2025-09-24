#include "DataFormats/Common/interface/Wrapper.h"

// Add includes for your classes here
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/MultiClusterToCaloParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociator.h"
#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"
#include "SimDataFormats/Associations/interface/TTTrackTruthPair.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociator.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociator.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

namespace {
  struct dictionary {
    // The produced types
    edm::Wrapper<edm::RefProd<std::vector<reco::PFCluster>>> w_refPF;
    edm::Wrapper<edm::RefProd<std::vector<SimCluster>>> w_refSim;

    edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<SimCluster>>, edm::RefProd<std::vector<reco::PFCluster>>>>
        w_keyVal1;
    edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<reco::PFCluster>>, edm::RefProd<std::vector<SimCluster>>>>
        w_keyVal2;

    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQualityGeneric<std::vector<SimCluster>,
                                                                      std::vector<reco::PFCluster>,
                                                                      std::pair<float, float>,
                                                                      unsigned int,
                                                                      edm::RefProd<std::vector<SimCluster>>,
                                                                      edm::RefProd<std::vector<reco::PFCluster>>,
                                                                      edm::Ref<std::vector<SimCluster>>,
                                                                      edm::Ref<std::vector<reco::PFCluster>>>>>
        w_assoc1;

    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQualityGeneric<std::vector<reco::PFCluster>,
                                                                      std::vector<SimCluster>,
                                                                      float,
                                                                      unsigned int,
                                                                      edm::RefProd<std::vector<reco::PFCluster>>,
                                                                      edm::RefProd<std::vector<SimCluster>>,
                                                                      edm::Ref<std::vector<reco::PFCluster>>,
                                                                      edm::Ref<std::vector<SimCluster>>>>>
        w_assoc2;
  };
}  // namespace