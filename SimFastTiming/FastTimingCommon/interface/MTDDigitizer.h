#ifndef FastTimingSimProducers_FastTimingCommon_MTDDigitizer_h
#define FastTimingSimProducers_FastTimingCommon_MTDDigitizer_h

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerBase.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTraits.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <tuple>


namespace mtd_digitizer {
  
  namespace MTDHelpers {
    // index , det id, time
    typedef std::tuple<int,uint32_t,float> MTDCaloHitTuple_t;
    
    bool orderByDetIdThenTime(const MTDCaloHitTuple_t &a, const MTDCaloHitTuple_t &b)
    {
      unsigned int detId_a(std::get<1>(a)), detId_b(std::get<1>(b));
      
      if(detId_a<detId_b) return true;
      if(detId_a>detId_b) return false;
      
      double time_a(std::get<2>(a)), time_b(std::get<2>(b));
      if(time_a<time_b) return true;
      
      return false;
    }
  }

  template<class Traits>
  class MTDDigitizer : public MTDDigitizerBase
  {
  public:

  typedef typename Traits::DeviceSim      DeviceSim ;
  typedef typename Traits::ElectronicsSim ElectronicsSim;
  typedef typename Traits::DigiCollection DigiCollection;

  MTDDigitizer(const edm::ParameterSet& config, 
	       edm::ConsumesCollector& iC,
	       edm::ProducerBase& parent) :
    MTDDigitizerBase(config,iC,parent),
    geom_(nullptr),
    deviceSim_( config.getParameterSet("DeviceSimulation") ),
    electronicsSim_( config.getParameterSet("ElectronicsSimulation") ),        
    maxSimHitsAccTime_( config.getParameter< uint32_t >("maxSimHitsAccTime") ) { }
    
    ~MTDDigitizer() override { }
    
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
      MTDSimHitDataAccumulator().swap(simHitAccumulator_);
    }
    
    const MTDGeometry* geom_;

    // implementations
    DeviceSim deviceSim_;           // processes a given simhit into an entry in a MTDSimHitDataAccumulator
    ElectronicsSim electronicsSim_; // processes a MTDSimHitDataAccumulator into a BTLDigiCollection/ETLDigiCollection
        
    //handle sim hits
    const int maxSimHitsAccTime_;
    MTDSimHitDataAccumulator simHitAccumulator_;  
        
  };

  template<class Traits>
  void MTDDigitizer<Traits>::accumulate(edm::Event const& e, 
					edm::EventSetup const& c, 
					CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits,0,hre);
  }

  template<class Traits>
  void MTDDigitizer<Traits>::accumulate(PileUpEventPrincipal const& e, 
					edm::EventSetup const& c, 
					CLHEP::HepRandomEngine* hre){
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits,e.bunchCrossing(),hre);
  }

  template<class Traits>
  void MTDDigitizer<Traits>::accumulate(edm::Handle<edm::PSimHitContainer> const &hits, 
					int bxCrossing, 
					CLHEP::HepRandomEngine* hre) {
    using namespace MTDHelpers;
    
    //create list of tuples (pos in container, RECO DetId, time) to be sorted first
    int nchits=(int)hits->size();  
    std::vector< MTDCaloHitTuple_t > hitRefs;
    hitRefs.reserve(nchits);
    for(int i=0; i<nchits; ++i) {
      const auto& the_hit = hits->at(i);    
      
      DetId id = the_hit.detUnitId();
      
      if (verbosity_>0) {	
	edm::LogInfo("MTDDigitizer") << " i/p " << std::hex << the_hit.detUnitId() << std::dec 
				     << " o/p " << id.rawId() << std::endl;
      }
      
      if( 0 != id.rawId() ) {      
	hitRefs.emplace_back( i, id.rawId(), the_hit.tof() );
      }
    }
    std::sort(hitRefs.begin(),hitRefs.end(),MTDHelpers::orderByDetIdThenTime);
    
    deviceSim_.getHitsResponse(hitRefs, hits, &simHitAccumulator_, hre);

    hitRefs.clear();

  }
  
  template<class Traits>
  void MTDDigitizer<Traits>::initializeEvent(edm::Event const& e, edm::EventSetup const& c) {
    deviceSim_.getEvent(e);
    electronicsSim_.getEvent(e);
  }
  
  template<class Traits>
  void MTDDigitizer<Traits>::finalizeEvent(edm::Event& e, edm::EventSetup const& c, 
					   CLHEP::HepRandomEngine* hre) {
    
    auto digiCollection = std::make_unique<DigiCollection>();
    electronicsSim_.run(simHitAccumulator_,*digiCollection, hre);
    e.put(std::move(digiCollection),digiCollection_);
    
    //release memory for next event
    resetSimHitDataAccumulator();
  }
    

  template<class Traits>
  void MTDDigitizer<Traits>::beginRun(const edm::EventSetup & es) {    

    edm::ESHandle<MTDGeometry> geom;
    es.get<MTDDigiGeometryRecord>().get(geom);
    geom_ = geom.product();

    deviceSim_.getEventSetup(es);
    electronicsSim_.getEventSetup(es);

  }
}

#endif
