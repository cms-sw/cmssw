#include "TrackingTools/TrackFitters/interface/KFFittingSmootherESProducer.h"
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

KFFittingSmootherESProducer::KFFittingSmootherESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

KFFittingSmootherESProducer::~KFFittingSmootherESProducer() {}

boost::shared_ptr<TrajectoryFitter> 
KFFittingSmootherESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_updator){
//     delete _updator;
//     _updator = 0;
//   }  
  std::string fname = pset_.getParameter<std::string>("Fitter");
  std::string sname = pset_.getParameter<std::string>("Smoother");
  double theEstimateCut = pset_.getParameter<double>("EstimateCut");
  int theMinNumberOfHits = pset_.getParameter<int>("MinNumberOfHits");

  edm::ESHandle<TrajectoryFitter> fit;
  edm::ESHandle<TrajectorySmoother> smooth;
  
  iRecord.get(fname, fit);
  iRecord.get(sname, smooth);
  
  _fitter  = boost::shared_ptr<TrajectoryFitter>(new KFFittingSmoother(*fit.product(), *smooth.product(),
								       theEstimateCut,theMinNumberOfHits));
  return _fitter;
}


