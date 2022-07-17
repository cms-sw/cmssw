// user includes
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/GroupedDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SimpleDAFHitCollector.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

// system includes
#include <string>
#include <memory>

class MultiRecHitCollectorESProducer : public edm::ESProducer {
public:
  MultiRecHitCollectorESProducer(const edm::ParameterSet& iConfig);
  ~MultiRecHitCollectorESProducer() override = default;
  std::unique_ptr<MultiRecHitCollector> produce(const MultiRecHitRecord&);

private:
  // es tokens
  edm::ESGetToken<SiTrackerMultiRecHitUpdator, MultiRecHitRecord> mrhToken;
  edm::ESGetToken<Propagator, CkfComponentsRecord> propAlongToken;
  edm::ESGetToken<Chi2MeasurementEstimatorBase, CkfComponentsRecord> chi2MeasToken;
  edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measToken;
  edm::ESGetToken<TrackerTopology, CkfComponentsRecord> topoToken;
  edm::ESGetToken<Propagator, CkfComponentsRecord> propOppositeToken;

  // configuration
  std::string myname_;
  std::string mrhupdator_;
  std::string propagatorAlongName_;
  std::string estimatorName_;
  std::string measurementTrackerName_;
  std::string mode_;
  bool debug_;
  std::string propagatorOppositeName_;
};

using namespace edm;

MultiRecHitCollectorESProducer::MultiRecHitCollectorESProducer(const edm::ParameterSet& iConfig)
    : myname_(iConfig.getParameter<std::string>("ComponentName")),
      mrhupdator_(iConfig.getParameter<std::string>("MultiRecHitUpdator")),
      propagatorAlongName_(iConfig.getParameter<std::string>("propagatorAlong")),
      estimatorName_(iConfig.getParameter<std::string>("estimator")),
      measurementTrackerName_(iConfig.getParameter<std::string>("MeasurementTrackerName")),
      mode_(iConfig.getParameter<std::string>("Mode")),
      debug_(iConfig.getParameter<bool>("Debug")),
      propagatorOppositeName_(iConfig.getParameter<std::string>("propagatorOpposite")) {
  auto cc = setWhatProduced(this, myname_);

  mrhToken = cc.consumes();
  propAlongToken = cc.consumes(edm::ESInputTag("", propagatorAlongName_));
  chi2MeasToken = cc.consumes(edm::ESInputTag("", estimatorName_));
  measToken = cc.consumes(edm::ESInputTag("", measurementTrackerName_));
  topoToken = cc.consumes();
  propOppositeToken = cc.consumes(edm::ESInputTag("", propagatorOppositeName_));
}

std::unique_ptr<MultiRecHitCollector> MultiRecHitCollectorESProducer::produce(const MultiRecHitRecord& iRecord) {
  const SiTrackerMultiRecHitUpdator* mrhu = &iRecord.get(mrhToken);
  const Propagator* propagator = &iRecord.get(propAlongToken);
  const Chi2MeasurementEstimatorBase* estimator = &iRecord.get(chi2MeasToken);
  const MeasurementTracker* measurement = &iRecord.get(measToken);
  const TrackerTopology* trackerTopology = &iRecord.get(topoToken);

  if (mode_ == "Grouped") {
    const Propagator* propagatorOpposite = &iRecord.get(propOppositeToken);

    return std::make_unique<GroupedDAFHitCollector>(
        measurement, mrhu, estimator, propagator, propagatorOpposite, debug_);
  } else {
    return std::make_unique<SimpleDAFHitCollector>(trackerTopology, measurement, mrhu, estimator, propagator, debug_);
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(MultiRecHitCollectorESProducer);
