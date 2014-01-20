#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimMuon/GEMDigitizer/interface/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModelFactory.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "CLHEP/Random/RandomEngine.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>


GEMDigiProducer::GEMDigiProducer(const edm::ParameterSet& ps)
  : collectionXF_(ps.getParameter<std::string>("inputCollection"))
  , digiModelString_(ps.getParameter<std::string>("digiModelString"))
{
  produces<GEMDigiCollection>();
  produces<StripDigiSimLinks>("GEM");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  gemDigiModel_ = GEMDigiModelFactory::get()->create("GEM" + digiModelString_ + "Model", ps);
  LogDebug("GEMDigiProducer") << "Using GEM" + digiModelString_ + "Model";

  gemDigiModel_->setRandomEngine(engine);
}


GEMDigiProducer::~GEMDigiProducer()
{
  delete gemDigiModel_;
}


void GEMDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // set geometry
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  gemDigiModel_->setGeometry(&*hGeom);
  gemDigiModel_->setup();

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByLabel("mix", collectionXF_, cf);

  std::auto_ptr<MixCollection<PSimHit> > hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output
  std::auto_ptr<GEMDigiCollection> digis(new GEMDigiCollection());
  std::auto_ptr<StripDigiSimLinks> stripDigiSimLinks(new StripDigiSimLinks() );

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
    
    gemDigiModel_->simulateSignal(roll, simHits);
    gemDigiModel_->simulateNoise(roll);
    gemDigiModel_->fillDigis(rawId, *digis);
    (*stripDigiSimLinks).insert(gemDigiModel_->stripDigiSimLinks());
  }
  
  // store them in the event
  e.put(digis);
  e.put(stripDigiSimLinks,"GEM");
}

