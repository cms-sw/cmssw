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

#include "SimMuon/GEMDigitizer/interface/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

GEMDigiProducer::GEMDigiProducer(const edm::ParameterSet& ps)
    : gemDigiModel_{
          GEMDigiModelFactory::get()->create("GEM" + ps.getParameter<std::string>("digiModelString") + "Model", ps)} {
  produces<GEMDigiCollection>();
  produces<StripDigiSimLinks>("GEM");
  produces<GEMDigiSimLinks>("GEM");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
        << "Add the service in the configuration file or remove the modules that require it.";
  }

  LogDebug("GEMDigiProducer") << "Using GEM" + ps.getParameter<std::string>("digiModelString") + "Model";

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
}

GEMDigiProducer::~GEMDigiProducer() = default;

void GEMDigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  gemDigiModel_->setGeometry(&*hGeom);
  gemDigiModel_->setup();
}

void GEMDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  MixCollection<PSimHit> hits{cf.product()};

  // Create empty output
  auto digis = std::make_unique<GEMDigiCollection>();
  auto stripDigiSimLinks = std::make_unique<StripDigiSimLinks>();
  auto gemDigiSimLinks = std::make_unique<GEMDigiSimLinks>();

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for (const auto& hit : hits) {
    hitMap[hit.detUnitId()].emplace_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto& etaPartitions(gemDigiModel_->getGeometry()->etaPartitions());

  for (const auto& roll : etaPartitions) {
    const GEMDetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto& simHits(hitMap[rawId]);

    LogDebug("GEMDigiProducer") << "GEMDigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;

    gemDigiModel_->simulateSignal(roll, simHits, engine);
    gemDigiModel_->simulateNoise(roll, engine);
    gemDigiModel_->fillDigis(rawId, *digis);
    (*stripDigiSimLinks).insert(gemDigiModel_->stripDigiSimLinks());
    (*gemDigiSimLinks).insert(gemDigiModel_->gemDigiSimLinks());
  }

  // store them in the event
  e.put(std::move(digis));
  e.put(std::move(stripDigiSimLinks), "GEM");
  e.put(std::move(gemDigiSimLinks), "GEM");
}
