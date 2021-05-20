#include "TrackingTools/RecoGeometry/plugins/GlobalDetLayerGeometryESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>
#include <string>

using namespace edm;

GlobalDetLayerGeometryESProducer::GlobalDetLayerGeometryESProducer(const edm::ParameterSet& p) {
  std::string myName = p.getParameter<std::string>("ComponentName");
  auto cc = setWhatProduced(this, myName);
  trackerToken_ = cc.consumes();
  muonToken_ = cc.consumes();
  mtdToken_ = cc.consumes();
}

GlobalDetLayerGeometryESProducer::~GlobalDetLayerGeometryESProducer() {}

std::unique_ptr<DetLayerGeometry> GlobalDetLayerGeometryESProducer::produce(const RecoGeometryRecord& iRecord) {
  auto const& tracker = iRecord.get(trackerToken_);
  auto const& muon = iRecord.get(muonToken_);
  edm::ESHandle<MTDDetLayerGeometry> mtd;

  // get the MTD if it is available
  if (auto mtdRecord = iRecord.tryToGetRecord<MTDRecoGeometryRecord>()) {
    mtd = mtdRecord->getHandle(mtdToken_);
    if (!mtd.isValid()) {
      LogInfo("GlobalDetLayergGeometryBuilder") << "No MTD geometry is available.";
    }
  } else {
    LogInfo("GlobalDetLayerGeometryBuilder") << "No MTDRecoGeometryRecord is available.";
  }

  // if we've got MTD initialize it
  if (mtd.isValid())
    return std::make_unique<GlobalDetLayerGeometry>(&tracker, &muon, mtd.product());

  return std::make_unique<GlobalDetLayerGeometry>(&tracker, &muon);
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalDetLayerGeometryESProducer);
