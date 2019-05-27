#include "SimMuon/GEMDigitizer/plugins/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/plugins/GEMDigiModule.h"

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
  : gemDigiModule_(std::make_unique<GEMDigiModule>(ps))
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

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
}


GEMDigiProducer::~GEMDigiProducer() = default;

void GEMDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>("inputCollection", "g4SimHitsMuonGEMHits");
  desc.add<std::string>("mixLabel", "mix");

  desc.add<double>("signalPropagationSpeed", 0.66);
  desc.add<double>("timeResolution", 5.);
  desc.add<double>("timeJitter", 1.0);
  desc.add<double>("averageShapingTime", 50.0);
  desc.add<double>("averageEfficiency", 0.98);
  desc.add<double>("averageNoiseRate", 0.001);
  
  desc.add<int>("minBunch", -5);
  desc.add<int>("maxBunch", 3);

  desc.add<bool>("fixedRollRadius", true);
  desc.add<bool>("digitizeOnlyMuons", false);
  desc.add<bool>("simulateBkgNoise", false);
  desc.add<bool>("simulateNoiseCLS", true);
  desc.add<bool>("simulateElectronBkg", false);
  desc.add<bool>("simulateIntrinsicNoise", false);

  desc.add<double>("instLumi", 7.5);
  desc.add<double>("rateFact", 1.0);
  desc.add<double>("referenceInstLumi", 5.);
  desc.add<double>("resolutionX", 0.03);
  
  desc.add<double>("GE11ModNeuBkgParam0", 5710.23);  
  desc.add<double>("GE11ModNeuBkgParam1", -43.3928); 
  desc.add<double>("GE11ModNeuBkgParam2", 0.0863681); 
  desc.add<double>("GE21ModNeuBkgParam0", 1440.44);
  desc.add<double>("GE21ModNeuBkgParam1", -7.48607); 
  desc.add<double>("GE21ModNeuBkgParam2", 0.0103078);
  desc.add<double>("GE11ElecBkgParam0", 406.249);
  desc.add<double>("GE11ElecBkgParam1", -2.90939);
  desc.add<double>("GE11ElecBkgParam2", 0.00548191);
  desc.add<double>("GE21ElecBkgParam0", 97.0505);
  desc.add<double>("GE21ElecBkgParam1", -43.3928);
  desc.add<double>("GE21ElecBkgParam2", 00.000550599);

  descriptions.add("simMuonGEMDigisDef",desc);
}

void GEMDigiProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  gemDigiModule_->setGeometry(&*hGeom);
  geometry_ = &*hGeom;
}


void GEMDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  std::unique_ptr<MixCollection<PSimHit> > hits(new MixCollection<PSimHit>(cf.product()));

  // Create empty output
  std::unique_ptr<GEMDigiCollection> digis(new GEMDigiCollection());
  std::unique_ptr<StripDigiSimLinks> stripDigiSimLinks(new StripDigiSimLinks() );
  std::unique_ptr<GEMDigiSimLinks> gemDigiSimLinks(new GEMDigiSimLinks() );

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for (const auto& hit: *hits){
    hitMap[hit.detUnitId()].emplace_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto & etaPartitions(geometry_->etaPartitions());

  for (const auto& roll: etaPartitions){
    const GEMDetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto & simHits(hitMap[rawId]);

    LogDebug("GEMDigiProducer")
      << "GEMDigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;

    gemDigiModule_->simulate(roll, simHits, engine);
    gemDigiModule_->fillDigis(rawId, *digis);
    (*stripDigiSimLinks).insert(gemDigiModule_->stripDigiSimLinks());
    (*gemDigiSimLinks).insert(gemDigiModule_->gemDigiSimLinks());
  }

  // store them in the event
  e.put(std::move(digis));
  e.put(std::move(stripDigiSimLinks),"GEM");
  e.put(std::move(gemDigiSimLinks),"GEM");
}

DEFINE_FWK_MODULE(GEMDigiProducer);
