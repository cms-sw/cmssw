#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoProducer.h"
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

ME0DigiPreRecoProducer::ME0DigiPreRecoProducer(const edm::ParameterSet& ps)
  : digiPreRecoModelString_(ps.getParameter<std::string>("digiPreRecoModelString"))
{
  produces<ME0DigiPreRecoCollection>();

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "ME0DigiPreRecoProducer::ME0PreRecoDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  me0DigiPreRecoModel_ = ME0DigiPreRecoModelFactory::get()->create("ME0" + digiPreRecoModelString_ + "Model", ps);
  LogDebug("ME0DigiPreRecoProducer") << "Using ME0" + digiPreRecoModelString_ + "Model";

  std::string mix_(ps.getParameter<std::string>("mixLabel"));
  std::string collection_(ps.getParameter<std::string>("inputCollection"));

  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_));
}


ME0DigiPreRecoProducer::~ME0DigiPreRecoProducer()
{
  delete me0DigiPreRecoModel_;
}


void ME0DigiPreRecoProducer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup)
{
  // set geometry
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  me0DigiPreRecoModel_->setGeometry(&*hGeom);
  me0DigiPreRecoModel_->setup();
}


void ME0DigiPreRecoProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByToken(cf_token, cf);

  std::auto_ptr<MixCollection<PSimHit> > hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output
  std::auto_ptr<ME0DigiPreRecoCollection> digis(new ME0DigiPreRecoCollection());

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for(auto &hit: *hits){
    hitMap[hit.detUnitId()].push_back(hit);
  }
  
  // simulate signal and noise for each eta partition
  const auto & etaPartitions(me0DigiPreRecoModel_->getGeometry()->etaPartitions());
  
  for(auto &roll: etaPartitions){
    const ME0DetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto & simHits(hitMap[rawId]);
    
    LogDebug("ME0DigiPreRecoProducer") 
      << "ME0DigiPreRecoProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;
    
    me0DigiPreRecoModel_->simulateSignal(roll, simHits, engine);
    me0DigiPreRecoModel_->simulateNoise(roll, engine);
    me0DigiPreRecoModel_->fillDigis(rawId, *digis);
  }
  
  // store them in the event
  e.put(digis);
}

