#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include <boost/shared_ptr.hpp>

namespace {

class  Chi2MeasurementEstimatorESProducer: public edm::ESProducer{
 public:
  Chi2MeasurementEstimatorESProducer(const edm::ParameterSet & p);
  virtual ~Chi2MeasurementEstimatorESProducer();
  boost::shared_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  boost::shared_ptr<Chi2MeasurementEstimatorBase> _estimator;
  edm::ParameterSet pset_;
};

Chi2MeasurementEstimatorESProducer::Chi2MeasurementEstimatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

Chi2MeasurementEstimatorESProducer::~Chi2MeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
Chi2MeasurementEstimatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
  double maxChi2 = pset_.getParameter<double>("MaxChi2");
  double nSigma = pset_.getParameter<double>("nSigma");
  
  _estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(new Chi2MeasurementEstimator(maxChi2,nSigma));
  return _estimator;
}

void Chi2MeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName","");
  desc.add<double>("MaxChi2",30);
  desc.add<double>("nSigma",3);
  descriptions.add("Chi2MeasurementEstimator", desc);
}


}



DEFINE_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);

