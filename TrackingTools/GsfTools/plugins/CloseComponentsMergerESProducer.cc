#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <iostream>
#include <memory>
#include <string>

/** Provides the "CloseComponents" algorithm ("Merger") for reducing 
 * the number of components in a multi-
 */

template <unsigned int N>
class CloseComponentsMergerESProducer : public edm::ESProducer {
public:
  CloseComponentsMergerESProducer(const edm::ParameterSet& p);
  ~CloseComponentsMergerESProducer() override;
  std::unique_ptr<MultiGaussianStateMerger<N> > produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const int maxComp_;
  const edm::ESGetToken<DistanceBetweenComponents<N>, TrackingComponentsRecord> distToken_;
};

#include "FWCore/Framework/interface/ModuleFactory.h"
typedef CloseComponentsMergerESProducer<5> CloseComponentsMergerESProducer5D;
DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsMergerESProducer5D);

template <unsigned int N>
CloseComponentsMergerESProducer<N>::CloseComponentsMergerESProducer(const edm::ParameterSet& p)
    : maxComp_(p.getParameter<int>("MaxComponents")),
      distToken_(setWhatProduced(this, p.getParameter<std::string>("ComponentName"))
                     .consumes(edm::ESInputTag("", p.getParameter<std::string>("DistanceMeasure")))) {}

template <unsigned int N>
CloseComponentsMergerESProducer<N>::~CloseComponentsMergerESProducer() {
  //   std::cout << "MultiGaussianState: "
  // 	    << MultiGaussianState<5>::instances_ << " "
  // 	    << MultiGaussianState<5>::maxInstances_ << " "
  // 	    << MultiGaussianState<5>::constructsCombinedState_ << std::endl;
  //   std::cout << "SingleGaussianState: "
  // 	    << SingleGaussianState<5>::instances_ << " "
  // 	    << SingleGaussianState<5>::maxInstances_ << " "
  // 	    << SingleGaussianState<5>::constructsWeightMatrix_ << std::endl;
  //   std::cout << "SingleGaussianState: "
  // 	    << SingleGaussianState<5>::instances_ << " "
  // 	    << SingleGaussianState<5>::maxInstances_ << " "
  // 	    << SingleGaussianState<5>::constructsWeightMatrix_ << std::endl;
  //   std::cout << "CloseComponentsMergerESProducer deleted" << std::endl;
}

template <unsigned int N>
typename std::unique_ptr<MultiGaussianStateMerger<N> > CloseComponentsMergerESProducer<N>::produce(
    const TrackingComponentsRecord& iRecord) {
  return std::unique_ptr<MultiGaussianStateMerger<N> >(
      new CloseComponentsMerger<N>(maxComp_, &iRecord.get(distToken_)));
}

template <unsigned int N>
void CloseComponentsMergerESProducer<N>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  desc.add<int>("MaxComponents");
  desc.add<std::string>("DistanceMeasure");

  descriptions.addDefault(desc);
}
