#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"

class PixelTrackFilterByKinematicsProducer : public edm::global::EDProducer<> {
public:
  explicit PixelTrackFilterByKinematicsProducer(const edm::ParameterSet& iConfig);
  ~PixelTrackFilterByKinematicsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const float theoPtMin;
  const float theNSigmaInvPtTolerance;
  const float theTIPMax;
  const float theNSigmaTipMaxTolerance;
  const float theChi2Max;
};

PixelTrackFilterByKinematicsProducer::PixelTrackFilterByKinematicsProducer(const edm::ParameterSet& iConfig)
    : theoPtMin(1 / iConfig.getParameter<double>("ptMin")),
      theNSigmaInvPtTolerance(iConfig.getParameter<double>("nSigmaInvPtTolerance")),
      theTIPMax(iConfig.getParameter<double>("tipMax")),
      theNSigmaTipMaxTolerance(iConfig.getParameter<double>("nSigmaTipMaxTolerance")),
      theChi2Max(iConfig.getParameter<double>("chi2")) {
  produces<PixelTrackFilter>();
}

void PixelTrackFilterByKinematicsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<double>("ptMin", 0.1);
  desc.add<double>("nSigmaInvPtTolerance", 0.0);
  desc.add<double>("tipMax", 1.0);
  desc.add<double>("nSigmaTipMaxTolerance", 0.0);
  desc.add<double>("chi2", 1000.0);

  descriptions.add("pixelTrackFilterByKinematics", desc);
}

PixelTrackFilterByKinematicsProducer::~PixelTrackFilterByKinematicsProducer() {}

void PixelTrackFilterByKinematicsProducer::produce(edm::StreamID,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  auto impl = std::make_unique<PixelTrackFilterByKinematics>(
      theoPtMin, theNSigmaInvPtTolerance, theTIPMax, theNSigmaTipMaxTolerance, theChi2Max);
  auto prod = std::make_unique<PixelTrackFilter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelTrackFilterByKinematicsProducer);
