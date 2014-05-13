#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include <boost/foreach.hpp>


//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps) :
  theHGCEEDigitizer_(ps),
  theHGCHEbackDigitizer_(ps),
  theHGCHEfrontDigitizer_(ps)
{
  //configure from cfg
  hitCollection_     = ps.getUntrackedParameter< std::string >("hitCollection");
  digiCollection_    = ps.getUntrackedParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getUntrackedParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getUntrackedParameter< int32_t >("bxTime");
  doTrivialDigis_    = ps.getUntrackedParameter< bool >("doTrivialDigis");

  //get the random number generator
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration") << "HGCDigitizer requires the RandomNumberGeneratorService - please add this service or remove the modules that require it";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theHGCEEDigitizer_.setRandomNumberEngine(engine);
  theHGCHEbackDigitizer_.setRandomNumberEngine(engine);
  theHGCHEfrontDigitizer_.setRandomNumberEngine(engine);
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es)
{
  resetSimHitDataAccumulator(); 
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es)
{
  if( producesEEDigis() ) 
    {
      std::auto_ptr<HGCEEDigiCollection> digiResult(new HGCEEDigiCollection() );
      theHGCEEDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEfrontDigis())
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEfrontDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEbackDigis() )
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEbackDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE back hits";
      e.put(digiResult,digiCollection());
    }
}

//
void HGCDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate in-time the main event
  accumulate(hits, 0);
}

//
void HGCDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate for the simulated bunch crossing
  accumulate(hits, e.bunchCrossing());
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing)
{
  for(edm::PCaloHitContainer::const_iterator hit_it = hits->begin(); hit_it != hits->end(); ++hit_it)
    {
      //for now use a single time sample
      int    itime = 0; //(int) ( hit_it->time() - bxTime_*bxCrossing ); // - jitter etc.;
      uint32_t id  = hit_it->id();
      double ien   = hit_it->energy();

      HGCSimHitDataAccumulator::iterator simHitIt=simHitAccumulator_.find(id);
      if(simHitIt==simHitAccumulator_.end())
	{
	  HGCSimHitData baseData(10,0);
	  simHitAccumulator_[id]=baseData;
	  simHitIt=simHitAccumulator_.find(id);
	}
      if(itime<0 || itime>(int)simHitIt->second.size()) continue;
      (simHitIt->second)[itime] += ien;
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
void HGCDigitizer::resetSimHitDataAccumulator()
{
  for( HGCSimHitDataAccumulator::iterator it = simHitAccumulator_.begin(); it!=simHitAccumulator_.end(); it++) 
    std::fill(it->second.begin(), it->second.end(),0.); 
}


//
HGCDigitizer::~HGCDigitizer()
{
}


