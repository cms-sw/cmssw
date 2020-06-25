#ifndef SimMuon_GEMDigitizer_ME0DigiPreRecoProducer_h
#define SimMuon_GEMDigitizer_ME0DigiPreRecoProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class ME0DigiPreRecoProducer : public edm::stream::EDProducer<> {
public:
  explicit ME0DigiPreRecoProducer(const edm::ParameterSet& ps);

  ~ME0DigiPreRecoProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //Name of Collection used for create the XF
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token;
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> geom_token_;

  std::string digiPreRecoModelString_;
  std::unique_ptr<ME0DigiPreRecoModel> me0DigiPreRecoModel_;
};

ME0DigiPreRecoProducer::ME0DigiPreRecoProducer(const edm::ParameterSet& ps)
    : digiPreRecoModelString_(ps.getParameter<std::string>("digiPreRecoModelString")),
      me0DigiPreRecoModel_{ME0DigiPreRecoModelFactory::get()->create("ME0" + digiPreRecoModelString_ + "Model", ps)} {
  produces<ME0DigiPreRecoCollection>();

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "ME0DigiPreRecoProducer::ME0PreRecoDigiProducer() - RandomNumberGeneratorService is not present in "
           "configuration file.\n"
        << "Add the service in the configuration file or remove the modules that require it.";
  }
  LogDebug("ME0DigiPreRecoProducer") << "Using ME0" + digiPreRecoModelString_ + "Model";

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
  geom_token_ = esConsumes<ME0Geometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

ME0DigiPreRecoProducer::~ME0DigiPreRecoProducer() = default;

void ME0DigiPreRecoProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  // set geometry
  edm::ESHandle<ME0Geometry> hGeom = eventSetup.getHandle(geom_token_);
  me0DigiPreRecoModel_->setGeometry(&*hGeom);
  me0DigiPreRecoModel_->setup();
}

void ME0DigiPreRecoProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  MixCollection<PSimHit> hits{cf.product()};

  // Create empty output
  auto digis = std::make_unique<ME0DigiPreRecoCollection>();

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for (const auto& hit : hits) {
    hitMap[hit.detUnitId()].push_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto& etaPartitions(me0DigiPreRecoModel_->getGeometry()->etaPartitions());

  for (const auto& roll : etaPartitions) {
    const ME0DetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto& simHits(hitMap[rawId]);

    LogDebug("ME0DigiPreRecoProducer") << "ME0DigiPreRecoProducer: found " << simHits.size()
                                       << " hit(s) in eta partition" << rawId;

    me0DigiPreRecoModel_->simulateSignal(roll, simHits, engine);
    me0DigiPreRecoModel_->simulateNoise(roll, engine);
    me0DigiPreRecoModel_->fillDigis(rawId, *digis);
  }

  // store them in the event
  e.put(std::move(digis));
}

DEFINE_FWK_MODULE(ME0DigiPreRecoProducer);
#endif
