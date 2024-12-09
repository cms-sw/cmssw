#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>

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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class CAParamsESProducer : public ESProducer {

  public:
    CAParamsESProducer(edm::ParameterSet const& iConfig);
    std::optional<reco::CAParamsHost> produce(const TrackerRecoGeometryRecord &iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:

    // Layers params
    // const std::vector<int> layerStarts_;
    const std::vector<double> caThetaCuts_;
    const std::vector<double> caDCACuts_;

    // Cells params
    const std::vector<int> pairGraph_;
    const std::vector<int> startingPairs_;
    const std::vector<int> phiCuts_;
    const std::vector<double> minZ_;
    const std::vector<double> maxZ_;
    const std::vector<double> maxR_;
    const double cellZ0Cut_;
    const double cellPtCut_;
    const bool doClusterCut_;
    const bool idealConditions_;

    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopologyToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeometryToken_;

  };

  CAParamsESProducer::CAParamsESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig),
        // layerStarts_(iConfig.getParameter<std::vector<int>>("layerStarts")),
        caThetaCuts_(iConfig.getParameter<std::vector<double>>("caThetaCuts")),
        caDCACuts_(iConfig.getParameter<std::vector<double>>("caDCACuts")),
        pairGraph_(iConfig.getParameter<std::vector<int>>("pairGraph")),
        startingPairs_(iConfig.getParameter<std::vector<int>>("startingPairs")),
        phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
        minZ_(iConfig.getParameter<std::vector<double>>("minZ")),
        maxZ_(iConfig.getParameter<std::vector<double>>("maxZ")),
        maxR_(iConfig.getParameter<std::vector<double>>("maxR")),
        cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
        cellPtCut_(iConfig.getParameter<double>("cellPtCut")),
        doClusterCut_(iConfig.getParameter<bool>("doClusterCut")),
        idealConditions_(iConfig.getParameter<bool>("idealConditions"))
 {
    auto cc = setWhatProduced(this);   
    tTopologyToken_ = cc.consumes();
    tGeometryToken_ = cc.consumes();
  }

  std::optional<reco::CAParamsHost> CAParamsESProducer::produce(
    const TrackerRecoGeometryRecord& iRecord) {

    assert(minZ_.size() == maxZ_.size()); 
    assert(minZ_.size() == maxR_.size()); 
    assert(minZ_.size() == phiCuts_.size()); 

    assert(caThetaCuts_.size() == caDCACuts_.size()); 
    
    int n_layers = caThetaCuts_.size(); //layerStarts_.size() - 1;

    int n_pairs = pairGraph_.size() / 2;

    assert(int(n_pairs) == int(minZ_.size())); 
    assert(*std::max_element(startingPairs_.begin(), startingPairs_.end()) < n_pairs);

    reco::CAParamsHost product{{{n_layers + 1,n_pairs}}, cms::alpakatools::host()};

    auto layerSoA = product.view();
    auto cellSoA = product.view<::reco::CACellsSoA>();

    std::cout << "No. Layers to be used = " << n_layers << std::endl;
    
    const auto &trackerTopology = &iRecord.get(tTopologyToken_);
    const auto &trackerGeometry = &iRecord.get(tGeometryToken_);
    
    // auto n_dets = trackerGeometry->detUnits().size();
    auto detUnits = trackerGeometry->detUnits();

    auto subSystem = 1;
    auto subSystemName = GeomDetEnumerators::tkDetEnum[subSystem];
    auto subSystemOffset = trackerGeometry->offsetDU(subSystemName);
    
    std::cout << "=========================================================================================================" << std::endl;
    std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
    auto oldLayer = 0u;
    auto layerCount = 0;
    subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
    subSystemOffset = trackerGeometry->offsetDU(subSystemName);

    std::vector<int> layerStarts(n_layers+1);

    for (auto& detUnit : detUnits) {
      unsigned int index = detUnit->index();

      if(index >= subSystemOffset)
      {
        subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
        subSystemOffset = trackerGeometry->offsetDU(subSystemName);
        std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
      }

      auto geoid = detUnit->geographicalId();
      auto layer = trackerTopology->layer(geoid);

      if(layer != oldLayer)
      {
          layerStarts[layerCount] = index;
          layerCount++;
          if (layerCount > n_layers + 1)
            break;
          oldLayer = layer;
          std::cout << " > New layer at module : " << index << " (detId: " <<  geoid << ")" << std::endl;
      }
            
    }

    std::cout << "=========================================================================================================" << std::endl;

    for (int i = 0; i < n_layers; ++i)
    {
      layerSoA.layerStarts()[i] = layerStarts[i];
      layerSoA.caThetaCut()[i] = caThetaCuts_[i];
      layerSoA.caDCACut()[i] = caDCACuts_[i];
      std::cout << i << " - > " << caDCACuts_[i] << " - " << layerStarts[i] << std::endl;
    }
    
    layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

    for (int i = 0; i < n_pairs; ++i)
    {
        cellSoA.graph()[i] = {{uint32_t(pairGraph_[2*i]),uint32_t(pairGraph_[2*i+1])}};
        cellSoA.phiCuts()[i] = phiCuts_[i];
        cellSoA.minz()[i] = minZ_[i];
        cellSoA.maxz()[i] = maxZ_[i];
        cellSoA.maxr()[i] = maxR_[i];
        cellSoA.startingPair()[i] = false;
    }
  
    for (const int& i : startingPairs_)
      cellSoA.startingPair()[i] = true;

    cellSoA.cellPtCut() = cellPtCut_;
    cellSoA.cellZ0Cut() = cellZ0Cut_;
    cellSoA.doClusterCut() = doClusterCut_;
    cellSoA.idealConditions() = idealConditions_;
    
    return product;

  }

  
  void CAParamsESProducer::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) {
    
    using namespace phase1PixelTopology;
    edm::ParameterSetDescription desc;

    // layers params
    // desc.add<std::vector<int>>("layerStarts", std::vector<int>(std::begin(layerStart), std::end(layerStart))) ->setComment("Layer module start.");
    desc.add<std::vector<double>>("caDCACuts", {0.15f,0.25f,0.25f,0.25f,0.25f,0.25f,0.25f,0.25f,0.25f,0.25f}) ->setComment("Cut on RZ alignement. One per layer, the layer being the midelle one for a triplet.");
    desc.add<std::vector<double>>("caThetaCuts", {0.002f,0.002f,0.002f,0.002f,0.003f,0.003f,0.003f,0.003f,0.003f,0.003f}) ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
    desc.add<std::vector<int>>("startingPairs",{0,1,2}) ->setComment("The list of the ids of pairs from which the CA ntuplets building may start."); //TODO could be parsed via an expression
    // cells params
    desc.add<std::vector<int>>("pairGraph", std::vector<int>(std::begin(layerPairs), std::end(layerPairs))) ->setComment("CA graph");
    desc.add<std::vector<int>>("phiCuts", std::vector<int>(std::begin(phicuts), std::end(phicuts))) ->setComment("Cuts in phi for cells");
    desc.add<std::vector<double>>("minZ", std::vector<double>(std::begin(minz), std::end(minz))) ->setComment("Cuts in min z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxZ", std::vector<double>(std::begin(maxz), std::end(maxz))) ->setComment("Cuts in max z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxR", std::vector<double>(std::begin(maxr), std::end(maxr))) ->setComment("Cuts in max r for cells");
    desc.add<double>("cellZ0Cut", 12.0f) ->setComment("Z0 cut for cells");
    desc.add<double>("cellPtCut", 0.5f) ->setComment("Preliminary pT cut on "); 
    desc.add<bool>("doClusterCut", true);
    desc.add<bool>("idealConditions", true);

    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAParamsESProducer);
