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

#include "MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl.h"

//
// Class declaration
//

class MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer : public edm::global::EDProducer<> {
public:
  explicit MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer(const edm::ParameterSet &);
  ~MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
  edm::EDGetTokenT<reco::SimToRecoCollectionMtd> simToRecoMap_;
  edm::EDGetTokenT<reco::RecoToSimCollectionMtd> recoToSimMap_;
};

MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer::
    MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer(const edm::ParameterSet &ps) {
  geomToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  topoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();

  simToRecoMap_ = consumes<reco::SimToRecoCollectionMtd>(ps.getParameter<edm::InputTag>("simToRecoMap"));
  recoToSimMap_ = consumes<reco::RecoToSimCollectionMtd>(ps.getParameter<edm::InputTag>("recoToSimMap"));

  // Register the product
  produces<reco::MtdRecoMergedClusterToSimMergedClusterAssociator>();
}

MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer::
    ~MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer() {}

void MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer::produce(edm::StreamID,
                                                                             edm::Event &iEvent,
                                                                             const edm::EventSetup &es) const {
  auto geometryHandle = es.getTransientHandle(geomToken_);
  const MTDGeometry *geom = geometryHandle.product();

  auto topologyHandle = es.getTransientHandle(topoToken_);
  const MTDTopology *topology = topologyHandle.product();

  mtd::MTDGeomUtil geomTools_;
  geomTools_.setGeometry(geom);
  geomTools_.setTopology(topology);

  auto simToRecoMapHandle = iEvent.getHandle(simToRecoMap_);
  reco::SimToRecoCollectionMtd simToRecoMap = *simToRecoMapHandle.product();
  auto recoToSimMapHandle = iEvent.getHandle(recoToSimMap_);
  reco::RecoToSimCollectionMtd recoToSimMap = *recoToSimMapHandle.product();

  auto impl = std::make_unique<MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl>(
      iEvent.productGetter(), geomTools_, simToRecoMap, recoToSimMap);
  auto toPut = std::make_unique<reco::MtdRecoMergedClusterToSimMergedClusterAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer::fillDescriptions(
    edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simToRecoMap", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation"));
  desc.add<edm::InputTag>("recoToSimMap", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation"));

  cfg.add("mtdRecoMergedClusterToSimMergedClusterAssociatorByHits", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsProducer);
