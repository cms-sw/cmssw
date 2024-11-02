#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAParamsSoACollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

// #include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// #include "Geometry/Records/interface/TrackerTopologyRcd.h"
// #include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// #include "MagneticField/Engine/interface/MagneticField.h"
// #include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
// #include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

// #include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
// #include "RecoLocalTracker/Records/interface/CAParamsRecord.h"
// #include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class CAParamsESProducer : public ESProducer {

  public:
    CAParamsESProducer(edm::ParameterSet const& iConfig);
    std::optional<reco::CAParamsHost> produce(const TrackerRecoGeometryRecord &iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:

    // Layers params
    const std::vector<int> layerStarts_;

    // Cells params
    const std::vector<int> pairGraph_;
    const std::vector<int> phiCuts_;
    const std::vector<double> minZ_;
    const std::vector<double> maxZ_;
    const std::vector<double> maxR_;
    const double cellZ0Cut_;
    const double cellPtCut_;
    const bool doClusterCut_;
    const bool idealConditions_;
    
    // Regions params
    const std::vector<int> regionStarts_;
    const std::vector<double> caThetaCut_;
    const std::vector<double> caDCACut_;

  };

  CAParamsESProducer::CAParamsESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig),
        layerStarts_(iConfig.getParameter<std::vector<int>>("layerStarts")),
        pairGraph_(iConfig.getParameter<std::vector<int>>("pairGraph")),
        phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
        minZ_(iConfig.getParameter<std::vector<double>>("minZ")),
        maxZ_(iConfig.getParameter<std::vector<double>>("maxZ")),
        maxR_(iConfig.getParameter<std::vector<double>>("maxR")),
        cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
        cellPtCut_(iConfig.getParameter<double>("cellPtCut")),
        doClusterCut_(iConfig.getParameter<bool>("doClusterCut")),
        idealConditions_(iConfig.getParameter<bool>("idealConditions")),
        regionStarts_(iConfig.getParameter<std::vector<int>>("regionStarts")),
        caThetaCut_(iConfig.getParameter<std::vector<double>>("caThetaCut")),
        caDCACut_(iConfig.getParameter<std::vector<double>>("caDCACut"))
 {
    setWhatProduced(this);    
  }

  std::optional<reco::CAParamsHost> CAParamsESProducer::produce(
    const TrackerRecoGeometryRecord& iRecord) {

    assert(minZ_.size() == maxZ_.size()); 
    assert(minZ_.size() == maxR_.size()); 
    assert(minZ_.size() == phiCuts_.size()); 

    assert(regionStarts_.size() == regionStarts_.size()); 
    assert(caThetaCut_.size() == caThetaCut_.size()); 
    assert(caDCACut_.size() == caDCACut_.size()); 
    
    int n_layers = layerStarts_.size();
    int n_pairs = pairGraph_.size() / 2;
    int n_regions = regionStarts_.size();

    assert(int(n_pairs) == int(minZ_.size())); 

    reco::CAParamsHost product{{{n_layers,n_pairs,n_regions}}, cms::alpakatools::host()};
    // auto product = std::make_unique<reco::CAParamsHost>({{n_layers,n_pairs,n_regions}},cms::alpakatools::host());

    auto layerSoA = product.view();
    auto cellSoA = product.view<::reco::CACellsSoA>();
    auto regionSoA = product.view<::reco::CARegionsSoA>();

    for (int i = 0; i < n_layers; ++i)
        layerSoA[i] = layerStarts_[i];
    
    for (int i = 0; i < n_pairs; ++i)
    {
        cellSoA.graph()[i] = {{uint32_t(pairGraph_[2*i]),uint32_t(pairGraph_[2*i+1])}};
        cellSoA.phiCuts()[i] = phiCuts_[i];
        cellSoA.minz()[i] = minZ_[i];
        cellSoA.maxz()[i] = maxZ_[i];
        cellSoA.maxr()[i] = maxR_[i];
    }
    cellSoA.cellPtCut() = cellPtCut_;
    cellSoA.cellZ0Cut() = cellZ0Cut_;
    cellSoA.doClusterCut() = doClusterCut_;
    cellSoA.idealConditions() = idealConditions_;
    
    for (int i = 0; i < n_regions; i++)
    {
        regionSoA.regionStarts()[i] = regionStarts_[i];
        regionSoA.caThetaCut()[i] = caThetaCut_[i];
        regionSoA.caDCACut()[i] = caDCACut_[i];
    }
    
    return product;

  }

  
  void CAParamsESProducer::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) {
    
    using namespace phase1PixelTopology;
    edm::ParameterSetDescription desc;

    // layers params
    desc.add<std::vector<int>>("layerStarts", std::vector<int>(std::begin(layerStart), std::end(layerStart))) ->setComment("Layer module start");
    // cells params
    desc.add<std::vector<int>>("pairGraph", std::vector<int>(std::begin(layerPairs), std::end(layerPairs))) ->setComment("CA graph");
    desc.add<std::vector<int>>("phiCuts", std::vector<int>(std::begin(phicuts), std::end(phicuts))) ->setComment("Cuts in phi for cells");
    desc.add<std::vector<double>>("minZ", std::vector<double>(std::begin(minz), std::end(minz))) ->setComment("Cuts in min z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxZ", std::vector<double>(std::begin(maxz), std::end(maxz))) ->setComment("Cuts in max z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxR", std::vector<double>(std::begin(maxr), std::end(maxr))) ->setComment("Cuts in max r for cells");
    desc.add<double>("cellZ0Cut", 12.0) ->setComment("Z0 cut for cells");
    desc.add<double>("cellPtCut", 0.5) ->setComment("Preliminary pT cut on "); 
    desc.add<bool>("doClusterCut", true);
    desc.add<bool>("idealConditions", true);
    // region params
    desc.add<std::vector<int>>("regionStarts", {0,100}) ->setComment("Layer module start");
    desc.add<std::vector<double>>("caThetaCut", {0,100}) ->setComment("Layer module start");
    desc.add<std::vector<double>>("caDCACut", {0,100}) ->setComment("Layer module start");



    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAParamsESProducer);