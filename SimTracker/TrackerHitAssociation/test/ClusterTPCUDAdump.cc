#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

class ClusterTPCUDAdump : public edm::global::EDAnalyzer<> {
public:
  using ClusterSLGPU = trackerHitAssociationHeterogeneous::ClusterSLView;
  using Clus2TP = ClusterSLGPU::Clus2TP;
  using ProductCUDA = trackerHitAssociationHeterogeneous::ProductCUDA;

  explicit ClusterTPCUDAdump(const edm::ParameterSet& iConfig);
  ~ClusterTPCUDAdump() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<cms::cuda::Product<ProductCUDA>> tokenGPU_;
};

ClusterTPCUDAdump::ClusterTPCUDAdump(const edm::ParameterSet& iConfig) : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
    tokenGPU_ = consumes<cms::cuda::Product<ProductCUDA>>(iConfig.getParameter<edm::InputTag>("clusterTP"));
  } else {
  }
}

void ClusterTPCUDAdump::analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const& hctp = iEvent.get(tokenGPU_);
    cms::cuda::ScopedContextProduce ctx{hctp};

    auto const& ctp = ctx.get(hctp);
    auto const& soa = ctp.view();
    assert(soa.links_d);
  } else {
  }
}

void ClusterTPCUDAdump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("clusterTP", edm::InputTag("tpClusterProducerCUDAPreSplitting"));
  descriptions.add("clusterTPCUDAdump", desc);
}

DEFINE_FWK_MODULE(ClusterTPCUDAdump);
