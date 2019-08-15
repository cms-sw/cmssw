#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"

#include <string>
#include <memory>

class TrackingRecHitPropagatorESProducer : public edm::ESProducer {
public:
  TrackingRecHitPropagatorESProducer(const edm::ParameterSet& p);

  std::unique_ptr<TrackingRecHitPropagator> produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
};

using namespace edm;

TrackingRecHitPropagatorESProducer::TrackingRecHitPropagatorESProducer(const edm::ParameterSet& p)
    : mfToken_(setWhatProduced(this, p.getParameter<std::string>("ComponentName"))
                   .consumesFrom<MagneticField, IdealMagneticFieldRecord>(
                       edm::ESInputTag("", p.getParameter<std::string>("SimpleMagneticField")))) {}

std::unique_ptr<TrackingRecHitPropagator> TrackingRecHitPropagatorESProducer::produce(
    const TrackingComponentsRecord& iRecord) {
  return std::make_unique<TrackingRecHitPropagator>(&iRecord.get(mfToken_));
}

void TrackingRecHitPropagatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  desc.add<std::string>("SimpleMagneticField", "");
  descriptions.addDefault(desc);
}
DEFINE_FWK_EVENTSETUP_MODULE(TrackingRecHitPropagatorESProducer);
