#ifndef TrackingTools_ESProducers_KFSwitching1DUpdatorESProducer_h
#define TrackingTools_ESProducers_KFSwitching1DUpdatorESProducer_h

/** KFSwitching1DUpdatorESProducer
 *  ESProducer for KFSwitching1DUpdator class.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include <memory>

class  KFSwitching1DUpdatorESProducer: public edm::ESProducer{
 public:
  KFSwitching1DUpdatorESProducer(const edm::ParameterSet & p);
  ~KFSwitching1DUpdatorESProducer() override; 
  std::shared_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord &);
 private:
  std::shared_ptr<TrajectoryStateUpdator> _updator;
  edm::ParameterSet pset_;
};


#endif




