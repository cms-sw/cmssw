#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/TkSeedGenerator/interface/ClusterChecker.h"

class ClusterCheckerEDProducer: public edm::stream::EDProducer<> {
public:

  ClusterCheckerEDProducer(const edm::ParameterSet& iConfig);
  ~ClusterCheckerEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  ClusterChecker theClusterCheck;
  bool theSilentOnClusterCheck;
};

ClusterCheckerEDProducer::ClusterCheckerEDProducer(const edm::ParameterSet& iConfig):
  theClusterCheck(iConfig, consumesCollector()),
  theSilentOnClusterCheck(iConfig.getUntrackedParameter<bool>("silentClusterCheck"))
{
  produces<bool>();
}

void ClusterCheckerEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  ClusterChecker::fillDescriptions(desc);
  desc.addUntracked<bool>("silentClusterCheck", false);

  descriptions.add("trackerClusterCheckDefault", desc);
}

void ClusterCheckerEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto ret = std::make_unique<bool>(true);

  //protection for big ass events...
  size_t clustsOrZero = theClusterCheck.tooManyClusters(iEvent);
  if (clustsOrZero){
    if (!theSilentOnClusterCheck)
	edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.";
    *ret = false;
  }

  iEvent.put(std::move(ret));
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterCheckerEDProducer);
