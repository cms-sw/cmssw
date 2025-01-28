#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <memory>
#include <string>

/** Provides algorithms to measure the distance between  components
 * (currently either using a Kullback-Leibler or a Mahalanobis distance)
 */

template <unsigned int N>
class DistanceBetweenComponentsESProducer : public edm::ESProducer {
public:
  DistanceBetweenComponentsESProducer(const edm::ParameterSet& p);

  std::unique_ptr<DistanceBetweenComponents<N> > produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool useKullbackLeibler_;
};

#include "FWCore/Framework/interface/ModuleFactory.h"
typedef DistanceBetweenComponentsESProducer<5> DistanceBetweenComponentsESProducer5D;
DEFINE_FWK_EVENTSETUP_MODULE(DistanceBetweenComponentsESProducer5D);

template <unsigned int N>
DistanceBetweenComponentsESProducer<N>::DistanceBetweenComponentsESProducer(const edm::ParameterSet& p)
    : useKullbackLeibler_(p.getParameter<std::string>("DistanceMeasure") == "KullbackLeibler") {
  std::string myname = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myname);
}

template <unsigned int N>
typename std::unique_ptr<DistanceBetweenComponents<N> > DistanceBetweenComponentsESProducer<N>::produce(
    const TrackingComponentsRecord& iRecord) {
  std::unique_ptr<DistanceBetweenComponents<N> > distance;
  if (useKullbackLeibler_)
    distance = std::unique_ptr<DistanceBetweenComponents<N> >(new KullbackLeiblerDistance<N>());
  // //   else if ( distName == "Mahalanobis" )
  // //     distance = std::unique_ptr<DistanceBetweenComponents>(new MahalanobisDistance());

  return distance;
}

template <unsigned int N>
void DistanceBetweenComponentsESProducer<N>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DistanceMeasure", "");
  desc.add<std::string>("ComponentName", "");
  descriptions.addWithDefaultLabel(desc);
}
