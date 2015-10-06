#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizer.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <boost/foreach.hpp>

//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector & iC) : 
  checkValidDetIds_(true),
  simHitAccumulator_( new HGCSimHitDataAccumulator ),
  mySubDet_(ForwardSubdetector::ForwardEmpty),
  refSpeed_(0.1*CLHEP::c_light) { //[CLHEP::c_light]=mm/ns convert to cm/ns
  
  //configure from cfg
  hitCollection_     = ps.getParameter< std::string >("hitCollection");
  digiCollection_    = ps.getParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getParameter< int32_t >("bxTime");
  digitizationType_  = ps.getParameter< uint32_t >("digitizationType");
  useAllChannels_    = ps.getParameter< bool >("useAllChannels");
  verbosity_         = ps.getUntrackedParameter< int32_t >("verbosity",0);
  tofDelay_          = ps.getParameter< double >("tofDelay");  

  iC.consumes<std::vector<PCaloHit> >(edm::InputTag("g4SimHits",hitCollection_));

  if(hitCollection_.find("HitsEE")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCEE;  
    theHGCEEDigitizer_=std::unique_ptr<HGCEEDigitizer>(new HGCEEDigitizer(ps) ); 
  }
  if(hitCollection_.find("HitsHEfront")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCHEF;
    theHGCHEfrontDigitizer_=std::unique_ptr<HGCHEfrontDigitizer>(new HGCHEfrontDigitizer(ps) );
  }
  if(hitCollection_.find("HitsHEback")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCHEB;
    theHGCHEbackDigitizer_=std::unique_ptr<HGCHEbackDigitizer>(new HGCHEbackDigitizer(ps) );
  }
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es) {
  resetSimHitDataAccumulator(); 
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es, CLHEP::HepRandomEngine* engine) {
  if( producesEEDigis() ) {
    std::auto_ptr<HGCEEDigiCollection> digiResult(new HGCEEDigiCollection() );
    theHGCEEDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_,engine);
    edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";
    e.put(digiResult,digiCollection());
  }
  if( producesHEfrontDigis()) {
    std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
    theHGCHEfrontDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_,engine);
    edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
    e.put(digiResult,digiCollection());
  }
  if( producesHEbackDigis() )  {
    std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
    theHGCHEbackDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_,engine);
    edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE back hits";
    e.put(digiResult,digiCollection());
  }
}

//
void HGCDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);

  //accumulate in-time the main event
  accumulate(hits, 0, geom, engine);
}

//
void HGCDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);

  //accumulate for the simulated bunch crossing
  accumulate(hits, e.bunchCrossing(), geom, engine);
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const edm::ESHandle<HGCalGeometry> &geom, CLHEP::HepRandomEngine* engine) {
  if(!geom.isValid()) return;
  const HGCalTopology &topo=geom->topology();
  const HGCalDDDConstants &dddConst=topo.dddConstants();

  for(edm::PCaloHitContainer::const_iterator hit_it = hits->begin(); hit_it != hits->end(); ++hit_it) {
    HGCalDetId simId( hit_it->id() );

    //skip this hit if after ganging it is not valid
    int layer(simId.layer()), cell(simId.cell());
    std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,topo.detectorType());
    cell  = recoLayerCell.first;
    layer = recoLayerCell.second;
    if(layer<0 || cell<0) continue;
     
    //assign the RECO DetId
    DetId id( producesEEDigis() ?
	      (uint32_t)HGCEEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell):
	      (uint32_t)HGCHEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell) );

    if (verbosity_>0) {
      if (producesEEDigis())
	std::cout << "HGCDigitizer: i/p " << simId << " o/p " << HGCEEDetId(id) << std::endl;
      else
	std::cout << "HGCDigitizer: i/p " << simId << " o/p " << HGCHEDetId(id) << std::endl;
    }

    //distance to the center of the detector
    float dist2center( geom->getPosition(id).mag() );

    //hit time: [time()]=ns  [centerDist]=cm [refSpeed_]=cm/ns + delay by 1ns
    //accumulate in 6 buckets of 25ns (4 pre-samples, 1 in-time, 1 post-sample)
    float tof(hit_it->time()-dist2center/refSpeed_+tofDelay_);
    int itime=floor( tof/bxTime_ ) ;
      
    //no need to add bx crossing - tof comes already corrected from the mixing module
    //itime += bxCrossing;
    itime += 4;
      
    if(itime<0 || itime>5) continue; 
      
    //energy deposited 
    HGCSimEn_t ien( hit_it->energy() );
      
    //check if already existing (perhaps could remove this in the future - 2nd event should have all defined)
    HGCSimHitDataAccumulator::iterator simHitIt=simHitAccumulator_->find(id);
    if(simHitIt==simHitAccumulator_->end()) {
      HGCSimHitData baseData;
      baseData.fill(0.);
      simHitAccumulator_->insert( std::make_pair(id,baseData) );
      simHitIt=simHitAccumulator_->find(id);
    }
      
    //check if time index is ok and store energy
    if(itime >= (int)simHitIt->second.size() ) continue;
    (simHitIt->second)[itime] += ien;
  }
  
  //add base data for noise simulation
  if(!checkValidDetIds_) return;
  if(!geom.isValid()) return;
  HGCSimHitData baseData;
  baseData.fill(0.);
  const std::vector<DetId> &validIds=geom->getValidDetIds(); 
  int nadded(0);
  if (useAllChannels_) {
    for(std::vector<DetId>::const_iterator it=validIds.begin(); it!=validIds.end(); it++) {
      uint32_t id(it->rawId());
      if(simHitAccumulator_->find(id)!=simHitAccumulator_->end()) continue;
      simHitAccumulator_->insert( std::make_pair(id,baseData) );
      nadded++;
    }
  }
  if (verbosity_ > 0) 
    std::cout << "HGCDigitizer:Added " << nadded << ":" << validIds.size() 
	      << " detIds without " << hitCollection_ 
	      << " in first event processed" << std::endl;
  checkValidDetIds_=false;
}

//
void HGCDigitizer::beginRun(const edm::EventSetup & es) { }

//
void HGCDigitizer::endRun() { }

//
void HGCDigitizer::resetSimHitDataAccumulator() {
  for( HGCSimHitDataAccumulator::iterator it = simHitAccumulator_->begin(); it!=simHitAccumulator_->end(); it++)  it->second.fill(0.);
}
