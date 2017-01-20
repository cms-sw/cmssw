#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

class CrossingFramePSimHitToPSimHitsConverter: public edm::global::EDProducer<> {
public:
  CrossingFramePSimHitToPSimHitsConverter(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  struct InputInfo {
    using Token = edm::EDGetTokenT<CrossingFrame<PSimHit> >;
    InputInfo(Token t, const std::string& i): token(t), instance(i) {}
    Token token;
    std::string instance;
  };

  std::vector<InputInfo> input_;
};

CrossingFramePSimHitToPSimHitsConverter::CrossingFramePSimHitToPSimHitsConverter(const edm::ParameterSet& iConfig) {
  auto src = iConfig.getParameter<std::vector<edm::InputTag> >("src");
  input_.reserve(src.size());
  for(const auto& tag: src) {
    input_.emplace_back(consumes<CrossingFrame<PSimHit> >(tag), tag.instance());
    produces<std::vector<PSimHit> >(input_.back().instance);
  }
}

void CrossingFramePSimHitToPSimHitsConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag> >("src", std::vector<edm::InputTag>());
  descriptions.add("crossingFramePSimHitToPSimHits", desc);
}

void CrossingFramePSimHitToPSimHitsConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  for(const auto& input: input_) {
    edm::Handle<CrossingFrame<PSimHit> > hframe;
    iEvent.getByToken(input.token, hframe);
    const auto& frame = *hframe;
    const auto& signalHits = frame.getSignal();
    const auto& pileupHits = frame.getPileups();

    auto output = std::make_unique<std::vector<PSimHit>>();
    output->reserve(signalHits.size() + pileupHits.size());
    for(const auto& ptr: signalHits)
      output->emplace_back(*ptr);
    for(const auto& ptr: pileupHits)
      output->emplace_back(*ptr);
    iEvent.put(std::move(output), input.instance);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CrossingFramePSimHitToPSimHitsConverter);
