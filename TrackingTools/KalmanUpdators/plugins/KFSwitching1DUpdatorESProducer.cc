/** KFSwitching1DUpdatorESProducer
 *  ESProducer for KFSwitching1DUpdator class.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include <memory>

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

class KFSwitching1DUpdatorESProducer : public edm::ESProducer {
public:
  KFSwitching1DUpdatorESProducer(const edm::ParameterSet& p);

  std::unique_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool doEndCap_;
};
using namespace edm;

KFSwitching1DUpdatorESProducer::KFSwitching1DUpdatorESProducer(const edm::ParameterSet& p)
    : doEndCap_(p.getParameter<bool>("doEndCap")) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myname);
}

std::unique_ptr<TrajectoryStateUpdator> KFSwitching1DUpdatorESProducer::produce(
    const TrackingComponentsRecord& iRecord) {
  return std::make_unique<KFSwitching1DUpdator>(doEndCap_);
}

void KFSwitching1DUpdatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("doEndCap");
  desc.add<std::string>("ComponentName");
  descriptions.addDefault(desc);
}
DEFINE_FWK_EVENTSETUP_MODULE(KFSwitching1DUpdatorESProducer);
