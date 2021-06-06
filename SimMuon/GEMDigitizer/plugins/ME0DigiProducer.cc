#ifndef SimMuon_GEMDigitizer_ME0DigiProducer_h
#define SimMuon_GEMDigitizer_ME0DigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GEMDigiSimLink/interface/ME0DigiSimLink.h"

#include "SimMuon/GEMDigitizer/interface/ME0DigiModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class ME0DigiProducer : public edm::stream::EDProducer<> {
public:
  typedef edm::DetSetVector<ME0DigiSimLink> ME0DigiSimLinks;

  explicit ME0DigiProducer(const edm::ParameterSet& ps);

  ~ME0DigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //Name of Collection used for create the XF
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token;
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> geom_token_;

  std::unique_ptr<ME0DigiModel> ME0DigiModel_;
};

ME0DigiProducer::ME0DigiProducer(const edm::ParameterSet& ps)
    : ME0DigiModel_{
          ME0DigiModelFactory::get()->create("ME0" + ps.getParameter<std::string>("digiModelString") + "Model", ps)} {
  produces<ME0DigiCollection>();
  produces<ME0DigiSimLinks>("ME0");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "ME0DigiProducer::ME0DigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
        << "Add the service in the configuration file or remove the modules that require it.";
  }

  LogDebug("ME0DigiProducer") << "Using ME0" + ps.getParameter<std::string>("digiModelString") + "Model";

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
  geom_token_ = esConsumes<ME0Geometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

ME0DigiProducer::~ME0DigiProducer() = default;

void ME0DigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  edm::ESHandle<ME0Geometry> hGeom = eventSetup.getHandle(geom_token_);
  ME0DigiModel_->setGeometry(&*hGeom);
  ME0DigiModel_->setup();
}

void ME0DigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  MixCollection<PSimHit> hits{cf.product()};

  // Create empty output
  auto digis = std::make_unique<ME0DigiCollection>();
  auto me0DigiSimLinks = std::make_unique<ME0DigiSimLinks>();

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for (const auto& hit : hits) {
    hitMap[hit.detUnitId()].emplace_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto& etaPartitions(ME0DigiModel_->getGeometry()->etaPartitions());

  for (const auto& roll : etaPartitions) {
    const ME0DetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto& simHits(hitMap[rawId]);

    LogDebug("ME0DigiProducer") << "ME0DigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;

    ME0DigiModel_->simulateSignal(roll, simHits, engine);
    ME0DigiModel_->simulateNoise(roll, engine);
    ME0DigiModel_->fillDigis(rawId, *digis);
    (*me0DigiSimLinks).insert(ME0DigiModel_->me0DigiSimLinks());
  }

  // store them in the event
  e.put(std::move(digis));
  e.put(std::move(me0DigiSimLinks), "ME0");
}

DEFINE_FWK_MODULE(ME0DigiProducer);
#endif
