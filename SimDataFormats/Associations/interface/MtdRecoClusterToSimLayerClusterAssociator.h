#ifndef SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociator_h
#define SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociator_h
// Original Author:  Martina Malberti

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// forward declarations

namespace reco {

  class MtdRecoClusterToSimLayerClusterAssociator {
  public:
    MtdRecoClusterToSimLayerClusterAssociator(std::unique_ptr<reco::MtdRecoClusterToSimLayerClusterAssociatorBaseImpl>);
    MtdRecoClusterToSimLayerClusterAssociator() = delete;
    explicit MtdRecoClusterToSimLayerClusterAssociator(edm::Uninitialized) noexcept {};
    MtdRecoClusterToSimLayerClusterAssociator(MtdRecoClusterToSimLayerClusterAssociator &&) = default;
    MtdRecoClusterToSimLayerClusterAssociator &operator=(MtdRecoClusterToSimLayerClusterAssociator &&) = default;
    MtdRecoClusterToSimLayerClusterAssociator(const MtdRecoClusterToSimLayerClusterAssociator &) =
        delete;  // stop default

    ~MtdRecoClusterToSimLayerClusterAssociator() = default;
    const MtdRecoClusterToSimLayerClusterAssociator &operator=(const MtdRecoClusterToSimLayerClusterAssociator &) =
        delete;  // stop default

    // ---------- const member functions ---------------------
    /// Associate RecoCluster to MtdSimLayerCluster
    reco::RecoToSimCollectionMtd associateRecoToSim(const edm::Handle<FTLClusterCollection> &btlRecoClusH,
                                                    const edm::Handle<FTLClusterCollection> &etlRecoClusH,
                                                    const edm::Handle<MtdSimLayerClusterCollection> &simClusH) const {
      return m_impl->associateRecoToSim(btlRecoClusH, etlRecoClusH, simClusH);
    };

    /// Associate MtdSimLayerCluster to RecoCluster
    reco::SimToRecoCollectionMtd associateSimToReco(const edm::Handle<FTLClusterCollection> &btlRecoClusH,
                                                    const edm::Handle<FTLClusterCollection> &etlRecoClusH,
                                                    const edm::Handle<MtdSimLayerClusterCollection> &simClusH) const {
      return m_impl->associateSimToReco(btlRecoClusH, etlRecoClusH, simClusH);
    };

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<MtdRecoClusterToSimLayerClusterAssociatorBaseImpl> m_impl;
  };
}  // namespace reco

#endif
