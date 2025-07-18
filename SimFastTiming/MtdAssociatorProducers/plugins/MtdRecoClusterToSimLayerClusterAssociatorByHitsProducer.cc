// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"

#include "MtdRecoClusterToSimLayerClusterAssociatorByHitsImpl.h"

//
// Class declaration
//

class MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer : public edm::global::EDProducer<> {
public:
  explicit MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer(const edm::ParameterSet &);
  ~MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  const double energyCut_;
  const double timeCut_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
};

MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer::MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer(
    const edm::ParameterSet &ps)
    : energyCut_(ps.getParameter<double>("energyCut")), timeCut_(ps.getParameter<double>("timeCut")) {
  geomToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  topoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();

  // Register the product
  produces<reco::MtdRecoClusterToSimLayerClusterAssociator>();
}

MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer::~MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer() {}

void MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer::produce(edm::StreamID,
                                                                      edm::Event &iEvent,
                                                                      const edm::EventSetup &es) const {
  auto geometryHandle = es.getTransientHandle(geomToken_);
  const MTDGeometry *geom = geometryHandle.product();

  auto topologyHandle = es.getTransientHandle(topoToken_);
  const MTDTopology *topology = topologyHandle.product();

  mtd::MTDGeomUtil geomTools_;
  geomTools_.setGeometry(geom);
  geomTools_.setTopology(topology);

  auto impl = std::make_unique<MtdRecoClusterToSimLayerClusterAssociatorByHitsImpl>(
      iEvent.productGetter(), energyCut_, timeCut_, geomTools_);
  auto toPut = std::make_unique<reco::MtdRecoClusterToSimLayerClusterAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<double>("energyCut", 5.);
  desc.add<double>("timeCut", 10.);

  cfg.add("mtdRecoClusterToSimLayerClusterAssociatorByHits", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MtdRecoClusterToSimLayerClusterAssociatorByHitsProducer);
