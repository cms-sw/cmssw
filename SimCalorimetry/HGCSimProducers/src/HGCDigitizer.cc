#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizer.h"
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
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps) :
  checkValidDetIds_(true),
  simHitAccumulator_( new HGCSimHitDataAccumulator ),
  mySubDet_(ForwardSubdetector::ForwardEmpty)
{
  //configure from cfg
  hitCollection_     = ps.getParameter< std::string >("hitCollection");
  digiCollection_    = ps.getParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getParameter< int32_t >("bxTime");
  digitizationType_  = ps.getParameter< uint32_t >("digitizationType");
  
  //get the random number generator
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration") << "HGCDigitizer requires the RandomNumberGeneratorService - please add this service or remove the modules that require it";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();
  if(hitCollection_.find("HitsEE")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCEE;  
    theHGCEEDigitizer_=std::unique_ptr<HGCEEDigitizer>(new HGCEEDigitizer(ps) ); 
    theHGCEEDigitizer_->setRandomNumberEngine(engine);
  }
  if(hitCollection_.find("HitsHEfront")!=std::string::npos)  
    { 
      mySubDet_=ForwardSubdetector::HGCHEF;
      theHGCHEfrontDigitizer_=std::unique_ptr<HGCHEfrontDigitizer>(new HGCHEfrontDigitizer(ps) );
      theHGCHEfrontDigitizer_->setRandomNumberEngine(engine);
    }
  if(hitCollection_.find("HitsHEback")!=std::string::npos)
    { 
      mySubDet_=ForwardSubdetector::HGCHEB;
      theHGCHEbackDigitizer_=std::unique_ptr<HGCHEbackDigitizer>(new HGCHEbackDigitizer(ps) );
      theHGCHEbackDigitizer_->setRandomNumberEngine(engine);
    }
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
      theHGCEEDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEfrontDigis())
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEfrontDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEbackDigis() )
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEbackDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_);
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

  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);

  //accumulate in-time the main event
  accumulate(hits, 0, geom);
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

  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);

  //accumulate for the simulated bunch crossing
  accumulate(hits, e.bunchCrossing(), geom);
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const edm::ESHandle<HGCalGeometry> &geom)
{
  for(edm::PCaloHitContainer::const_iterator hit_it = hits->begin(); hit_it != hits->end(); ++hit_it)
    {
      HGCalDetId simId( hit_it->id() );

      //gang SIM->RECO cells
      int layer(simId.layer()), cell(simId.cell());
      float zPos(0.);
      if(geom.isValid())
	{
	  const HGCalTopology &topo=geom->topology();
	  const HGCalDDDConstants &dddConst=topo.dddConstants();
	  zPos=dddConst.getFirstTrForm()->h3v.z();
	  std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,topo.detectorType());
	  cell  = recoLayerCell.first;
	  layer = recoLayerCell.second;
	  if(layer<0) continue;
	}

      //assign the RECO DetId
      DetId id( producesEEDigis() ?
		(uint32_t)HGCEEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell):
		(uint32_t)HGCHEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell) );

      //hit time: [time()]=ns  [zPos]=cm [CLHEP::c_light]=mm/ns
      //for now accumulate in buckets of bxTime_
      int itime=floor( (hit_it->time()-zPos/(0.1*CLHEP::c_light))/bxTime_);
      itime += bxCrossing;
      if(itime<0) continue;
      
      //energy deposited 
      HGCSimEn_t ien( hit_it->energy() );
      
      //check if already existing (perhaps could remove this in the future - 2nd event should have all defined)
      HGCSimHitDataAccumulator::iterator simHitIt=simHitAccumulator_->find(id);
      if(simHitIt==simHitAccumulator_->end())
	{
	  HGCSimHitData baseData;
	  baseData.fill(0.);
	  simHitAccumulator_->insert( std::make_pair(id,baseData) );
	  simHitIt=simHitAccumulator_->find(id);
	}
      
      //check if time is ok
      if( itime >= (int)(simHitIt->second.size()) ) continue;
      
      (simHitIt->second)[itime] += ien;
    }
  
  //add base data for noise simulation
  if(!checkValidDetIds_) return;
  if(!geom.isValid()) return;
  HGCSimHitData baseData;
  baseData.fill(0.);
  const std::vector<DetId> &validIds=geom->getValidDetIds(); 
  int nadded(0);
  for(std::vector<DetId>::const_iterator it=validIds.begin(); it!=validIds.end(); it++)
    {
      uint32_t id(it->rawId());
      if(simHitAccumulator_->find(id)!=simHitAccumulator_->end()) continue;
      simHitAccumulator_->insert( std::make_pair(id,baseData) );
      nadded++;
    }
  std::cout << "Added " << nadded << " detIds without " << hitCollection_ << " in first event processed" << std::endl;
  checkValidDetIds_=false;
}

//
void HGCDigitizer::beginRun(const edm::EventSetup & es)
{
}

//
void HGCDigitizer::endRun()
{
}

//
void HGCDigitizer::resetSimHitDataAccumulator()
{
  for( HGCSimHitDataAccumulator::iterator it = simHitAccumulator_->begin(); it!=simHitAccumulator_->end(); it++)  it->second.fill(0.);
}





