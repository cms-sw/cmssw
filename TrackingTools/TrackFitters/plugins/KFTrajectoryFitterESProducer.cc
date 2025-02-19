#include "TrackingTools/TrackFitters/plugins/KFTrajectoryFitterESProducer.h"
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

KFTrajectoryFitterESProducer::KFTrajectoryFitterESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

KFTrajectoryFitterESProducer::~KFTrajectoryFitterESProducer() {}

boost::shared_ptr<TrajectoryFitter> 
KFTrajectoryFitterESProducer::produce(const TrajectoryFitterRecord & iRecord){ 

  std::string pname = pset_.getParameter<std::string>("Propagator");
  std::string uname = pset_.getParameter<std::string>("Updator");
  std::string ename = pset_.getParameter<std::string>("Estimator");
  std::string gname = pset_.getParameter<std::string>("RecoGeometry");
  int minHits = pset_.getParameter<int>("minHits");

  edm::ESHandle<Propagator> prop;
  edm::ESHandle<TrajectoryStateUpdator> upd;
  edm::ESHandle<Chi2MeasurementEstimatorBase> est;
  edm::ESHandle<DetLayerGeometry> geo;

  iRecord.getRecord<TrackingComponentsRecord>().get(pname, prop);
  iRecord.getRecord<TrackingComponentsRecord>().get(uname, upd);
  iRecord.getRecord<TrackingComponentsRecord>().get(ename, est);
  iRecord.getRecord<RecoGeometryRecord>().get(gname,geo);

  _fitter  = boost::shared_ptr<TrajectoryFitter>(new KFTrajectoryFitter(prop.product(),
									upd.product(),
									est.product(),
									minHits,
  									geo.product() ));
  return _fitter;
}


