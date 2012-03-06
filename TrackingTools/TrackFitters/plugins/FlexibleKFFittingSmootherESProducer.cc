#include "TrackingTools/TrackFitters/plugins/FlexibleKFFittingSmootherESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

FlexibleKFFittingSmootherESProducer::FlexibleKFFittingSmootherESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

FlexibleKFFittingSmootherESProducer::~FlexibleKFFittingSmootherESProducer() {}

boost::shared_ptr<TrajectoryFitter> 
FlexibleKFFittingSmootherESProducer::produce(const TrajectoryFitterRecord & iRecord){ 

  std::string standardFitterName = pset_.getParameter<std::string>("standardFitter");
  std::string looperFitterName = pset_.getParameter<std::string>("looperFitter");

  edm::ESHandle<TrajectoryFitter> standardFitter;
  edm::ESHandle<TrajectoryFitter> looperFitter;
  
  iRecord.get(standardFitterName,standardFitter);
  iRecord.get(looperFitterName,looperFitter);
  
  _fitter  = boost::shared_ptr<TrajectoryFitter>(new FlexibleKFFittingSmoother(*standardFitter.product(),
									       *looperFitter.product()   ) );
  return _fitter;
}


