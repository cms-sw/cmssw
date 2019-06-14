#ifndef TrackingTools_ESProducers_KFUpdatorESProducer_h
#define TrackingTools_ESProducers_KFUpdatorESProducer_h

/** KFUpdatorESProducer
 *  ESProducer for KFUpdator class.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include <memory>

class KFUpdatorESProducer : public edm::ESProducer {
public:
  KFUpdatorESProducer(const edm::ParameterSet &p);
  ~KFUpdatorESProducer() override;
  std::unique_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord &);

private:
  edm::ParameterSet pset_;
};

#endif
