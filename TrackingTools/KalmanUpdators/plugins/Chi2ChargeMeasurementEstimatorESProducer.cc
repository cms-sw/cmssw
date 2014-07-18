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
}

Chi2ChargeMeasurementEstimatorESProducer::~Chi2ChargeMeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
Chi2ChargeMeasurementEstimatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  double maxChi2 = pset_.getParameter<double>("MaxChi2");
  double nSigma = pset_.getParameter<double>("nSigma");
  bool cutOnPixelCharge_ = pset_.exists("minGoodPixelCharge");
  bool cutOnStripCharge_ = pset_.exists("minGoodStripCharge");
  double minGoodPixelCharge_= (cutOnPixelCharge_ ? pset_.getParameter<double>("minGoodPixelCharge") : 0); 
  double minGoodStripCharge_= (cutOnStripCharge_ ? pset_.getParameter<double>("minGoodStripCharge") : 0);
  double pTChargeCutThreshold_= (pset_.exists("pTChargeCutThreshold") ? pset_.getParameter<double>("pTChargeCutThreshold") : -1.);
  
  _estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(
	new Chi2ChargeMeasurementEstimator(maxChi2,nSigma, cutOnPixelCharge_, cutOnStripCharge_, 
		minGoodPixelCharge_, minGoodStripCharge_, pTChargeCutThreshold_));
  return _estimator;
}


