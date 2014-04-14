#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <boost/foreach.hpp>

//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps)
{
  //configure from cfg
  hitCollection_     = iConfig.getUntrackedParameter< std::string >("hitCollection");
  maxSimHitsAccTime_ = iConfig.getUntrackedParameter< uint32_t >("maxSimHitsAcctime");

  //get the random number generator
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration") << "HGCDigitizer requires the RandomNumberGeneratorService - please add this service or remove the modules that require it";
  }
  //CLHEP::HepRandomEngine& engine = rng->getEngine();
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es)
{
  resetSimHitDataAccumulator(); 
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es)
{
  //if(theHBHEDigitizer){
    //std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
    //theHBHEDigitizer->run(*hbheResult);
    //edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << hbheResult->size() <<  " EE hits";
    //e.put(hbheResult);
  //}
}

//
void HGCDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  iEvent.getByLabel(edm::InputTag("g4SimHits",hitCollection_),caloHits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate in-time the main event
  accumulate(hits, 0);
}

//
void HcalDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  iEvent.getByLabel(edm::InputTag("g4SimHits",hitCollection_),caloHits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate for the simulated bunch crossing
  accumulate(hits, e.bunchCrossing());
}

//
void HcalDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing)
{
  for(edm::PCaloHitContainer::const_iterator hit_it = hits->begin(); hit_it != hits->end(); ++hit_it)
    {
      uint32_t id  = hit_it->id();
      double ien   = hit_it->energy();
      int    itime = (int) hit_it->time(); // - 25*bxCrossing - jitter etc.;

      CaloSimHitDataAccumulator simHitIt=simHitAccumulator_.find(id);
      if(simHitIt==simHitAccumulator_.end())
	{
	  CaloSimHitData baseData(25,0);
	  simHitAccumulator_[id]=baseData;
	  simHitIt=simHitAccumulator_.find(id);
	}
      if(itime<0 || itime>simHitIt->size()) continue;
      (*simHit)[itime] += ien;

    }
}

//
void HGCDigitizer::beginRun(const edm::EventSetup & es)
{
  //checkGeometry(es);
  //theShapes->beginRun(es);
}

//
void HGCDigitizer::endRun()
{
  //theShapes->endRun();   
}

//
HGCDigitizer::~HGCDigitizer()
{
}


