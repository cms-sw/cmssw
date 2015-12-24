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
  boost::shared_ptr<Chi2MeasurementEstimatorBase> m_estimator;
  edm::ParameterSet const m_pset;
};

Chi2MeasurementEstimatorESProducer::Chi2MeasurementEstimatorESProducer(const edm::ParameterSet & p) :
  m_pset(p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this,myname);
}

Chi2MeasurementEstimatorESProducer::~Chi2MeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
Chi2MeasurementEstimatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
  auto maxChi2 = m_pset.getParameter<double>("MaxChi2");
  auto nSigma  = m_pset.getParameter<double>("nSigma");
  auto maxSag  = m_pset.existsAs<double>("MaxSagita") ? m_pset.getParameter<double>("MaxSagita") : -1.;
  auto minToll = m_pset.existsAs<double>("MinimalTollerance") ?  m_pset.getParameter<double>("MinimalTollerance") : 10;
   
  m_estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(new Chi2MeasurementEstimator(maxChi2,nSigma, maxSag, minToll));
  return m_estimator;
}

void Chi2MeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName","Chi2");
  desc.add<double>("MaxChi2",30);
  desc.add<double>("nSigma",3);
  desc.add<double>("MaxSagita",2.);
  desc.add<double>("MinimalTollerance",0.5);
  descriptions.add("Chi2MeasurementEstimator", desc);
}


}



DEFINE_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);

