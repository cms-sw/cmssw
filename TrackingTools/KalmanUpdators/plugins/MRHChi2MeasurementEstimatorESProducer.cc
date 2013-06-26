#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimatorESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <string>
#include <memory>

using namespace edm;

MRHChi2MeasurementEstimatorESProducer::MRHChi2MeasurementEstimatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

MRHChi2MeasurementEstimatorESProducer::~MRHChi2MeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
MRHChi2MeasurementEstimatorESProducer::produce(const TrackingComponentsRecord& iRecord){ 
  
  double maxChi2 = pset_.getParameter<double>("MaxChi2");
  double nSigma = pset_.getParameter<double>("nSigma");
  _estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(new MRHChi2MeasurementEstimator(maxChi2,nSigma));
  return _estimator;
}


