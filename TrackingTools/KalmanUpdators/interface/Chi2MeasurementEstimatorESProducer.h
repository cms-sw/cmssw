#ifndef TrackingTools_ESProducers_Chi2MeasurementEstimatorESProducer_h
#define TrackingTools_ESProducers_Chi2MeasurementEstimatorESProducer_h

/** \class Chi2MeasurementEstimatorESProducer
 *  ESProducer for Chi2MeasurementEstimator.
 *
 *  $Date: 2007/05/09 13:58:19 $
 *  $Revision: 1.1.2.1 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include <boost/shared_ptr.hpp>

class  Chi2MeasurementEstimatorESProducer: public edm::ESProducer{
 public:
  Chi2MeasurementEstimatorESProducer(const edm::ParameterSet & p);
  virtual ~Chi2MeasurementEstimatorESProducer(); 
  boost::shared_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Chi2MeasurementEstimatorBase> _estimator;
  edm::ParameterSet pset_;
};


#endif




