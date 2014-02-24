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

#include "SimMuon/GEMDigitizer/interface/ME0DigiProducer.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiModelFactory.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "CLHEP/Random/RandomEngine.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>


ME0DigiProducer::ME0DigiProducer(const edm::ParameterSet& ps)
  : collectionXF_(ps.getParameter<std::string>("inputCollection"))
  , digiModelString_(ps.getParameter<std::string>("digiModelString"))
{
  produces<ME0DigiCollection>();
  produces<StripDigiSimLinks>("ME0");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "ME0DigiProducer::ME0DigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  me0DigiModel_ = ME0DigiModelFactory::get()->create("ME0" + digiModelString_ + "Model", ps);
  LogDebug("ME0DigiProducer") << "Using ME0" + digiModelString_ + "Model";

  me0DigiModel_->setRandomEngine(engine);
}


ME0DigiProducer::~ME0DigiProducer()
{
  delete me0DigiModel_;
}


void ME0DigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // set geometry
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  me0DigiModel_->setGeometry(&*hGeom);
  me0DigiModel_->setup();

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByLabel("mix", collectionXF_, cf);

  std::auto_ptr<MixCollection<PSimHit> > hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output
  std::auto_ptr<ME0DigiCollection> digis(new ME0DigiCollection());
  std::auto_ptr<StripDigiSimLinks> stripDigiSimLinks(new StripDigiSimLinks() );

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for(auto &hit: *hits){
    hitMap[hit.detUnitId()].push_back(hit);
  }
  
  // simulate signal and noise for each eta partition
  const auto & etaPartitions(me0DigiModel_->getGeometry()->etaPartitions());

  for(auto &roll: etaPartitions){
    const ME0DetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto & simHits(hitMap[rawId]);
    
    LogDebug("ME0DigiProducer") 
      << "ME0DigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;
    
    me0DigiModel_->simulateSignal(roll, simHits);
    me0DigiModel_->simulateNoise(roll);
    me0DigiModel_->fillDigis(rawId, *digis);
    (*stripDigiSimLinks).insert(me0DigiModel_->stripDigiSimLinks());
  }
  
  // store them in the event
  e.put(digis);
  e.put(stripDigiSimLinks,"ME0");
}

