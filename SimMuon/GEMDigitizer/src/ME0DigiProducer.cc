#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimMuon/GEMDigitizer/interface/ME0DigiProducer.h"
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

ME0DigiProducer::ME0DigiProducer(const edm::ParameterSet& ps)
    : ME0DigiModel_{
          ME0DigiModelFactory::get()->create("ME0" + ps.getParameter<std::string>("digiModelString") + "Model", ps)} {
  produces<ME0DigiCollection>();
  produces<StripDigiSimLinks>("ME0");
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
}

ME0DigiProducer::~ME0DigiProducer() = default;

void ME0DigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
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
  auto stripDigiSimLinks = std::make_unique<StripDigiSimLinks>();
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
    (*stripDigiSimLinks).insert(ME0DigiModel_->stripDigiSimLinks());
    (*me0DigiSimLinks).insert(ME0DigiModel_->me0DigiSimLinks());
  }

  // store them in the event
  e.put(std::move(digis));
  e.put(std::move(stripDigiSimLinks), "ME0");
  e.put(std::move(me0DigiSimLinks), "ME0");
}
