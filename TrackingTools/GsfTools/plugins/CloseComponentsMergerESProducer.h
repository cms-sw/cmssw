#ifndef CloseComponentsMergerESProducer_h_
#define CloseComponentsMergerESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include <memory>

/** Provides the "CloseComponents" algorithm ("Merger") for reducing 
 * the number of components in a multi-
 */

template <unsigned int N>
class CloseComponentsMergerESProducer : public edm::ESProducer {
public:
  CloseComponentsMergerESProducer(const edm::ParameterSet &p);
  ~CloseComponentsMergerESProducer() override;
  std::unique_ptr<MultiGaussianStateMerger<N> > produce(const TrackingComponentsRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const int maxComp_;
  const edm::ESGetToken<DistanceBetweenComponents<N>, TrackingComponentsRecord> distToken_;
};

#include "TrackingTools/GsfTools/plugins/CloseComponentsMergerESProducer.icc"

#endif
