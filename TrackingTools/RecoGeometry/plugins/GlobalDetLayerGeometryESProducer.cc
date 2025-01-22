#include <memory>
#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"

class GlobalDetLayerGeometryESProducer : public edm::ESProducer {
public:
  GlobalDetLayerGeometryESProducer(const edm::ParameterSet& p);
  ~GlobalDetLayerGeometryESProducer() override = default;
  std::unique_ptr<DetLayerGeometry> produce(const RecoGeometryRecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;
  edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonToken_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> mtdToken_;
};

using namespace edm;

GlobalDetLayerGeometryESProducer::GlobalDetLayerGeometryESProducer(const edm::ParameterSet& p) {
  std::string myName = p.getParameter<std::string>("ComponentName");
  auto cc = setWhatProduced(this, myName);
  trackerToken_ = cc.consumes();
  muonToken_ = cc.consumes();
  mtdToken_ = cc.consumes();
}

void GlobalDetLayerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", std::string(""));
  descriptions.addWithDefaultLabel(desc);
}

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
