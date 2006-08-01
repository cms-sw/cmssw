#ifndef CloseComponentsTSOSMergerESProducer_h_
#define CloseComponentsTSOSMergerESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include <boost/shared_ptr.hpp>

/** Provides the "CloseComponents" algorithm ("Merger") for reducing 
 * the number of components in a multi-TSOS
 */

class  CloseComponentsTSOSMergerESProducer: public edm::ESProducer{
 public:
  CloseComponentsTSOSMergerESProducer(const edm::ParameterSet & p);
  virtual ~CloseComponentsTSOSMergerESProducer(); 
  boost::shared_ptr<MultiTrajectoryStateMerger> produce(const TrackingComponentsRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




