#include "TrackingTools/KalmanUpdators/interface/Chi2ChargeMeasurementEstimatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

Chi2ChargeMeasurementEstimatorESProducer::Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);

  maxChi2_             = pset_.getParameter<double>("MaxChi2");
  nSigma_              = pset_.getParameter<double>("nSigma");
  cutOnPixelCharge_    = pset_.exists("minGoodPixelCharge");
  cutOnStripCharge_    = pset_.exists("minGoodStripCharge");
  minGoodPixelCharge_  = (cutOnPixelCharge_ ? pset_.getParameter<double>("minGoodPixelCharge") : 0); 
  minGoodStripCharge_  = (cutOnStripCharge_ ? pset_.getParameter<double>("minGoodStripCharge") : 0);
  pTChargeCutThreshold_= pset_.getParameter<double>("pTChargeCutThreshold");
  

}

Chi2ChargeMeasurementEstimatorESProducer::~Chi2ChargeMeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
Chi2ChargeMeasurementEstimatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  _estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(
	new Chi2ChargeMeasurementEstimator(maxChi2_,nSigma_, cutOnPixelCharge_, cutOnStripCharge_, 
		minGoodPixelCharge_, minGoodStripCharge_, pTChargeCutThreshold_));
  return _estimator;
}


