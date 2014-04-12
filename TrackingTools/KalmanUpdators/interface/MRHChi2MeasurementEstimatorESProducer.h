#ifndef TrackingTools_ESProducers_MRHChi2MeasurementEstimatorESProducer_h
#define TrackingTools_ESProducers_MRHChi2MeasurementEstimatorESProducer_h


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimator.h"
#include <boost/shared_ptr.hpp>

class  MRHChi2MeasurementEstimatorESProducer: public edm::ESProducer{
 public:
  MRHChi2MeasurementEstimatorESProducer(const edm::ParameterSet & p);
  virtual ~MRHChi2MeasurementEstimatorESProducer(); 
  boost::shared_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);
 private:
  boost::shared_ptr<Chi2MeasurementEstimatorBase> _estimator;
  edm::ParameterSet pset_;
};


#endif




