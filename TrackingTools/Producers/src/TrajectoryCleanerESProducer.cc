#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerFactory.h"

class TrajectoryCleanerESProducer : public edm::ESProducer {
public:
  TrajectoryCleanerESProducer(const edm::ParameterSet&);
  ~TrajectoryCleanerESProducer() override;

  typedef std::unique_ptr<TrajectoryCleaner> ReturnType;

  ReturnType produce(const TrackingComponentsRecord&);

private:
  std::string theComponentName;
  std::string theComponentType;
  edm::ParameterSet theConfig;
};

TrajectoryCleanerESProducer::TrajectoryCleanerESProducer(const edm::ParameterSet& iConfig) {
  theComponentName = iConfig.getParameter<std::string>("ComponentName");
  theComponentType = iConfig.getParameter<std::string>("ComponentType");

  theConfig = iConfig;
  setWhatProduced(this, theComponentName);
}

TrajectoryCleanerESProducer::~TrajectoryCleanerESProducer() {}

// ------------ method called to produce the data  ------------
TrajectoryCleanerESProducer::ReturnType TrajectoryCleanerESProducer::produce(const TrackingComponentsRecord& iRecord) {
  using namespace edm::es;

  return ReturnType(TrajectoryCleanerFactory::get()->create(theComponentType, theConfig));
}

DEFINE_FWK_EVENTSETUP_MODULE(TrajectoryCleanerESProducer);
