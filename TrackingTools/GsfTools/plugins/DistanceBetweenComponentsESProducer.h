#ifndef DistanceBetweenComponentsESProducer_h_
#define DistanceBetweenComponentsESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include <boost/shared_ptr.hpp>

/** Provides algorithms to measure the distance between  components
 * (currently either using a Kullback-Leibler or a Mahalanobis distance)
 */

template <unsigned int N>
class  DistanceBetweenComponentsESProducer : public edm::ESProducer{
 public:
  DistanceBetweenComponentsESProducer(const edm::ParameterSet & p);
  ~DistanceBetweenComponentsESProducer() override; 
  std::unique_ptr< DistanceBetweenComponents<N> > produce(const TrackingComponentsRecord &);
 private:
  edm::ParameterSet pset_;
};

#include "TrackingTools/GsfTools/plugins/DistanceBetweenComponentsESProducer.icc"

#endif
