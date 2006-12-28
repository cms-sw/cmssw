#include "TrackingTools/TrackFitters/interface/KFTrajectorySmootherESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include <string>
#include <memory>

using namespace edm;

KFTrajectorySmootherESProducer::KFTrajectorySmootherESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

KFTrajectorySmootherESProducer::~KFTrajectorySmootherESProducer() {}

boost::shared_ptr<TrajectorySmoother> 
KFTrajectorySmootherESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  std::string pname = pset_.getParameter<std::string>("Propagator");
  std::string uname = pset_.getParameter<std::string>("Updator");
  std::string ename = pset_.getParameter<std::string>("Estimator");

  edm::ESHandle<Propagator> prop;
  edm::ESHandle<TrajectoryStateUpdator> upd;
  edm::ESHandle<Chi2MeasurementEstimatorBase> est;

  iRecord.get(pname, prop);
  iRecord.get(uname, upd);
  iRecord.get(ename, est);

  _smoother  = boost::shared_ptr<TrajectorySmoother>(new KFTrajectorySmoother(prop.product(),
									      upd.product(),
									      est.product()));
  return _smoother;
}


