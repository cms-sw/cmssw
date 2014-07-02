#ifndef TrackingTools_ESProducers_Chi2ChargeMeasurementEstimatorESProducer_h
#define TrackingTools_ESProducers_Chi2ChargeMeasurementEstimatorESProducer_h

/** \class Chi2ChargeMeasurementEstimatorESProducer
 *  ESProducer for Chi2ChargeMeasurementEstimator.
 *
 *  \author speer
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2ChargeMeasurementEstimator.h"
#include <boost/shared_ptr.hpp>

class  Chi2ChargeMeasurementEstimatorESProducer: public edm::ESProducer{
 public:
  Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet & p);
  virtual ~Chi2ChargeMeasurementEstimatorESProducer(); 
  boost::shared_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Chi2MeasurementEstimatorBase> _estimator;
  edm::ParameterSet pset_;
};


#endif




