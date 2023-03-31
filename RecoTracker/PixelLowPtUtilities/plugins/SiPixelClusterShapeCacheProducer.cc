#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterData.h"

#include <cassert>

class SiPixelClusterShapeCacheProducer : public edm::global::EDProducer<> {
public:
  explicit SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig);
  ~SiPixelClusterShapeCacheProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  using InputCollection = edmNew::DetSetVector<SiPixelCluster>;

  const edm::EDGetTokenT<InputCollection> token_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
};

SiPixelClusterShapeCacheProducer::SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig)
    : token_(consumes<InputCollection>(iConfig.getParameter<edm::InputTag>("src"))), geomToken_(esConsumes()) {
  if (iConfig.getParameter<bool>("onDemand")) {
    throw cms::Exception("OnDemandNotAllowed")
        << "Use of the `onDemand` feature of SiPixelClusterShapeCacheProducer is no longer supported";
  }
  produces<SiPixelClusterShapeCache>();
}

SiPixelClusterShapeCacheProducer::~SiPixelClusterShapeCacheProducer() {}

void SiPixelClusterShapeCacheProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClusters"));
  desc.add<bool>("onDemand", false)->setComment("The on demand feature is no longer supported");
  descriptions.add("siPixelClusterShapeCache", desc);
}

void SiPixelClusterShapeCacheProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<InputCollection> input;
  iEvent.getByToken(token_, input);

  if (!input.isValid()) {
    edm::LogError("siPixelClusterShapeCache") << "input pixel cluster collection is not valid!";
    auto output = std::make_unique<SiPixelClusterShapeCache>();
    iEvent.put(std::move(output));
    return;
  }

  const auto& geom = &iSetup.getData(geomToken_);

  auto output = std::make_unique<SiPixelClusterShapeCache>(input);
  output->resize(input->data().size());

  ClusterData data;  // reused
  ClusterShape clusterShape;

  for (const auto& detSet : *input) {
    const GeomDetUnit* genericDet = geom->idToDetUnit(detSet.detId());
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);

    edmNew::DetSet<SiPixelCluster>::const_iterator iCluster = detSet.begin(), endCluster = detSet.end();
    for (; iCluster != endCluster; ++iCluster) {
      SiPixelClusterShapeCache::ClusterRef clusterRef = edmNew::makeRefTo(input, iCluster);
      if (not output->isFilled(clusterRef)) {
        data.size.clear();
        clusterShape.determineShape(*pixDet, *iCluster, data);
        output->insert(clusterRef, data);
      }
    }
  }
  output->shrink_to_fit();

  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(SiPixelClusterShapeCacheProducer);
