#ifndef SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociator_h
#define SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociator_h

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl.h"

// forward declarations

namespace reco {

  class MtdRecoMergedClusterToSimMergedClusterAssociator {
  public:
    MtdRecoMergedClusterToSimMergedClusterAssociator(
        std::unique_ptr<reco::MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl>);
    MtdRecoMergedClusterToSimMergedClusterAssociator() = default;
    MtdRecoMergedClusterToSimMergedClusterAssociator(MtdRecoMergedClusterToSimMergedClusterAssociator &&) = default;
    MtdRecoMergedClusterToSimMergedClusterAssociator &operator=(MtdRecoMergedClusterToSimMergedClusterAssociator &&) =
        default;
    MtdRecoMergedClusterToSimMergedClusterAssociator(const MtdRecoMergedClusterToSimMergedClusterAssociator &) =
        delete;  // stop default

    ~MtdRecoMergedClusterToSimMergedClusterAssociator() = default;
    const MtdRecoMergedClusterToSimMergedClusterAssociator &operator=(
        const MtdRecoMergedClusterToSimMergedClusterAssociator &) = delete;  // stop default

    // ---------- const member functions ---------------------
    /// Associate RecoMergedCluster to MtdSimMergedCluster
    reco::MergedRecoToSimCollectionMtd associateRecoToSim(
        const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const {
      return m_impl->associateRecoToSim(btlRecoClusH, etlRecoClusH, simClusH);
    };

    /// Associate MtdSimMergedCluster to RecoMergedCluster
    reco::MergedSimToRecoCollectionMtd associateSimToReco(
        const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const {
      return m_impl->associateSimToReco(btlRecoClusH, etlRecoClusH, simClusH);
    };

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl> m_impl;
  };
}  // namespace reco

#endif
