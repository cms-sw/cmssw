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
  : digiModelString_(ps.getParameter<std::string>("digiModelString"))
{
  produces<GEMDigiCollection>();
  produces<StripDigiSimLinks>("GEM");
  produces<GEMDigiSimLinks>("GEM");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  gemDigiModel_ = GEMDigiModelFactory::get()->create("GEM" + digiModelString_ + "Model", ps);
  LogDebug("GEMDigiProducer") << "Using GEM" + digiModelString_ + "Model";

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
}


GEMDigiProducer::~GEMDigiProducer()
{
  delete gemDigiModel_;
}


void GEMDigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  gemDigiModel_->setGeometry(&*hGeom);
  gemDigiModel_->setup();
}


void GEMDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  std::auto_ptr<MixCollection<PSimHit> > hits(new MixCollection<PSimHit>(cf.product()));

  // Create empty output
  std::auto_ptr<GEMDigiCollection> digis(new GEMDigiCollection());
  std::auto_ptr<StripDigiSimLinks> stripDigiSimLinks(new StripDigiSimLinks() );
  std::auto_ptr<GEMDigiSimLinks> gemDigiSimLinks(new GEMDigiSimLinks() );

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for(auto &hit: *hits){
    hitMap[hit.detUnitId()].push_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto & etaPartitions(gemDigiModel_->getGeometry()->etaPartitions());

  for(auto &roll: etaPartitions){
    const GEMDetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto & simHits(hitMap[rawId]);

    LogDebug("GEMDigiProducer")
      << "GEMDigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;

    gemDigiModel_->simulateSignal(roll, simHits, engine);
    gemDigiModel_->simulateNoise(roll, engine);
    gemDigiModel_->fillDigis(rawId, *digis);
    (*stripDigiSimLinks).insert(gemDigiModel_->stripDigiSimLinks());
    (*gemDigiSimLinks).insert(gemDigiModel_->gemDigiSimLinks());
  }

  // store them in the event
  e.put(digis);
  e.put(stripDigiSimLinks,"GEM");
  e.put(gemDigiSimLinks,"GEM");
}

