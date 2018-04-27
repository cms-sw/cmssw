#ifndef FastTimingSimProducers_FastTimingCommon_FTLDigitizer_h
#define FastTimingSimProducers_FastTimingCommon_FTLDigitizer_h

#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerBase.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <tuple>


namespace ftl_digitizer {
  
  namespace FTLHelpers {
    // index , det id, time
    typedef std::tuple<int,uint32_t,float> FTLCaloHitTuple_t;
    
    bool orderByDetIdThenTime(const FTLCaloHitTuple_t &a, const FTLCaloHitTuple_t &b)
    {
      unsigned int detId_a(std::get<1>(a)), detId_b(std::get<1>(b));
      
      if(detId_a<detId_b) return true;
      if(detId_a>detId_b) return false;
      
      double time_a(std::get<2>(a)), time_b(std::get<2>(b));
      if(time_a<time_b) return true;
      
      return false;
    }
  }

  template<class SensorPhysics, class ElectronicsSim>
  class FTLDigitizer : public FTLDigitizerBase
  {
  public:
    
  FTLDigitizer(const edm::ParameterSet& config, 
	       edm::ConsumesCollector& iC,
	       edm::ProducerBase& parent) :
    FTLDigitizerBase(config,iC,parent),
    deviceSim_( config.getParameterSet("DeviceSimulation") ),
    electronicsSim_( config.getParameterSet("ElectronicsSimulation") ),        
    maxSimHitsAccTime_( config.getParameter< uint32_t >("maxSimHitsAccTime") ),
    bxTime_( config.getParameter< double >("bxTime") ),         
    tofDelay_( config.getParameter< double >("tofDelay") ) { }
    
    ~FTLDigitizer() override { }
    
    /**
       @short handle SimHit accumulation
    */
    void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(edm::Handle<edm::PSimHitContainer> const &hits, int bxCrossing, CLHEP::HepRandomEngine* hre) override;
    
    /**
       @short actions at the start/end of event
    */
    void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    
    /**
       @short actions at the start/end of run
    */
    void beginRun(const edm::EventSetup & es) override; 
    void endRun() override {}
    
  private :
    
    void resetSimHitDataAccumulator() {
      FTLSimHitDataAccumulator().swap(simHitAccumulator_);
    }
    
    // implementations
    SensorPhysics deviceSim_;       // processes a given simhit into an entry in a FTLSimHitDataAccumulator
    ElectronicsSim electronicsSim_; // processes a FTLSimHitDataAccumulator into a BTLDigiCollection/ETLDigiCollection
        
    //handle sim hits
    const int maxSimHitsAccTime_;
    const double bxTime_;
    FTLSimHitDataAccumulator simHitAccumulator_;  
        
    //delay to apply after evaluating time of arrival at the sensitive detector
    const float tofDelay_;

  };

  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::accumulate(edm::Event const& e, 
							      edm::EventSetup const& c, 
							      CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits,0,hre);
  }

  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::accumulate(PileUpEventPrincipal const& e, 
							      edm::EventSetup const& c, 
							      CLHEP::HepRandomEngine* hre){
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits,e.bunchCrossing(),hre);
  }

  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::accumulate(edm::Handle<edm::PSimHitContainer> const &hits, 
							      int bxCrossing, 
							      CLHEP::HepRandomEngine* hre) {
    using namespace FTLHelpers;
    
    //create list of tuples (pos in container, RECO DetId, time) to be sorted first
    int nchits=(int)hits->size();  
    std::vector< FTLCaloHitTuple_t > hitRefs;
    hitRefs.reserve(nchits);
    for(int i=0; i<nchits; ++i) {
      const auto& the_hit = hits->at(i);    
      
      DetId id = the_hit.detUnitId();
      
      if (verbosity_>0) {	
	edm::LogInfo("FTLDigitizer") << " i/p " << std::hex << the_hit.detUnitId() << std::dec 
				     << " o/p " << id.rawId() << std::endl;
      }
      
      if( 0 != id.rawId() ) {      
	hitRefs.emplace_back( i, id.rawId(), the_hit.tof() );
      }
    }
    std::sort(hitRefs.begin(),hitRefs.end(),FTLHelpers::orderByDetIdThenTime);
    
    deviceSim_.getHitsResponse(hitRefs, hits, bxTime_, tofDelay_, &simHitAccumulator_);

    hitRefs.clear();

  }
  
  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::initializeEvent(edm::Event const& e, edm::EventSetup const& c) {
    deviceSim_.getEvent(e);
    electronicsSim_.getEvent(e);
  }
  
  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::finalizeEvent(edm::Event& e, edm::EventSetup const& c, 
								 CLHEP::HepRandomEngine* hre) {

    if ( isBTL() ){
      
      auto digiCollection = std::make_unique<BTLDigiCollection>();
      electronicsSim_.runBTL(simHitAccumulator_,*digiCollection);
      e.put(std::move(digiCollection),digiCollection_);
    
    }
    else {
      
      auto digiCollection = std::make_unique<ETLDigiCollection>();
      electronicsSim_.runETL(simHitAccumulator_,*digiCollection);
      e.put(std::move(digiCollection),digiCollection_);
    
    }

    //release memory for next event
    resetSimHitDataAccumulator();
  }
    

  template<class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics,ElectronicsSim>::beginRun(const edm::EventSetup & es) {
    deviceSim_.getEventSetup(es);
    electronicsSim_.getEventSetup(es);
  }
}


#endif

