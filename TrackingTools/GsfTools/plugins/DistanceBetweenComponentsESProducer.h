#ifndef DistanceBetweenComponentsESProducer_h_
#define DistanceBetweenComponentsESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include <memory>

/** Provides algorithms to measure the distance between  components
 * (currently either using a Kullback-Leibler or a Mahalanobis distance)
 */

template <unsigned int N>
class DistanceBetweenComponentsESProducer : public edm::ESProducer {
public:
  DistanceBetweenComponentsESProducer(const edm::ParameterSet &p);

  std::unique_ptr<DistanceBetweenComponents<N> > produce(const TrackingComponentsRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const bool useKullbackLeibler_;
};

#include "TrackingTools/GsfTools/plugins/DistanceBetweenComponentsESProducer.icc"

#endif
