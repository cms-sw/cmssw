#ifndef LargestWeightsTSOSMergerESProducer_h_
#define LargestWeightsTSOSMergerESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include <boost/shared_ptr.hpp>

/** Provides the "LargestWeights" algorithm ("Merger") for reducing 
 * the number of components in a multi-TSOS
 */

class  LargestWeightsTSOSMergerESProducer: public edm::ESProducer{
 public:
  LargestWeightsTSOSMergerESProducer(const edm::ParameterSet & p);
  virtual ~LargestWeightsTSOSMergerESProducer(); 
  boost::shared_ptr<MultiTrajectoryStateMerger> produce(const TrackingComponentsRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




