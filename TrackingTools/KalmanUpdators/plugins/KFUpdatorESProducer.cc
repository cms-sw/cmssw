#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include <memory>

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;
class KFUpdatorESProducer : public edm::ESProducer {
public:
  KFUpdatorESProducer(const edm::ParameterSet& p);
  std::unique_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
};

KFUpdatorESProducer::KFUpdatorESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myname);
}

std::unique_ptr<TrajectoryStateUpdator> KFUpdatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  return std::make_unique<KFUpdator>();
}

void KFUpdatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(KFUpdatorESProducer);
