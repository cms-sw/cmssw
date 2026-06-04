#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl.h"

namespace reco {
  MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl::MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl() {
  }
  MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl::~MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl() {
  }

  reco::MergedRecoToSimCollectionMtd MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
      const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const {
    return reco::MergedRecoToSimCollectionMtd();
  }

  reco::MergedSimToRecoCollectionMtd MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
      const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const {
    return reco::MergedSimToRecoCollectionMtd();
  }

}  // namespace reco
