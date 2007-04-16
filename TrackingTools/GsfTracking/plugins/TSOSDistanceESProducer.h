#ifndef TSOSDistanceESProducer_h_
#define TSOSDistanceESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"
#include <boost/shared_ptr.hpp>

/** Provides algorithms to measure the distance between TSOS components
 * (currently either using a Kullback-Leibler or a Mahalanobis distance)
 */

class  TSOSDistanceESProducer: public edm::ESProducer{
 public:
  TSOSDistanceESProducer(const edm::ParameterSet & p);
  virtual ~TSOSDistanceESProducer(); 
  boost::shared_ptr<TSOSDistanceBetweenComponents> produce(const TrackingComponentsRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




