#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizer.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include <algorithm>
#include <boost/foreach.hpp>

using namespace hgc_digi;

//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps,
                           edm::ConsumesCollector& iC) :
  checkValidDetIds_(true),
  simHitAccumulator_( new HGCSimHitDataAccumulator ),
  mySubDet_(ForwardSubdetector::ForwardEmpty),
  refSpeed_(0.1*CLHEP::c_light) //[CLHEP::c_light]=mm/ns convert to cm/ns
{
  //configure from cfg
  hitCollection_     = ps.getParameter< std::string >("hitCollection");
  digiCollection_    = ps.getParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getParameter< double >("bxTime");
  digitizationType_  = ps.getParameter< uint32_t >("digitizationType");
  useAllChannels_    = ps.getParameter< bool >("useAllChannels");
  verbosity_         = ps.getUntrackedParameter< uint32_t >("verbosity",0);
  tofDelay_          = ps.getParameter< double >("tofDelay");  

  iC.consumes<std::vector<PCaloHit> >(edm::InputTag("g4SimHits",hitCollection_));
  
  if(hitCollection_.find("HitsEE")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCEE;  
    theHGCEEDigitizer_=std::unique_ptr<HGCEEDigitizer>(new HGCEEDigitizer(ps) ); 
  }
  if(hitCollection_.find("HitsHEfront")!=std::string::npos)  
    { 
      mySubDet_=ForwardSubdetector::HGCHEF;
      theHGCHEfrontDigitizer_=std::unique_ptr<HGCHEfrontDigitizer>(new HGCHEfrontDigitizer(ps) );
    }
  if(hitCollection_.find("HitsHEback")!=std::string::npos)
    { 
      mySubDet_=ForwardSubdetector::HGCHEB;
      theHGCHEbackDigitizer_=std::unique_ptr<HGCHEbackDigitizer>(new HGCHEbackDigitizer(ps) );
    }
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es) 
{
  resetSimHitDataAccumulator(); 
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es, CLHEP::HepRandomEngine* hre)
{
  if( producesEEDigis() ) 
    {
      std::unique_ptr<HGCEEDigiCollection> digiResult(new HGCEEDigiCollection() );
      theHGCEEDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";      
      e.put(std::move(digiResult),digiCollection());
    }
  if( producesHEfrontDigis())
    {
      std::unique_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEfrontDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
      e.put(std::move(digiResult),digiCollection());
    }
  if( producesHEbackDigis() )
    {
      std::unique_ptr<HGCBHDigiCollection> digiResult(new HGCBHDigiCollection() );
      theHGCHEbackDigitizer_->run(digiResult,*simHitAccumulator_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE back hits";
      e.put(std::move(digiResult),digiCollection());
    }
}

//
void HGCDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* hre) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //get geometry
  edm::ESHandle<CaloGeometry> geom;
  eventSetup.get<CaloGeometryRecord>().get(geom);
  
  const HGCalGeometry* gHGCal = nullptr;
  const HcalGeometry* gHcal = nullptr;
  if( producesEEDigis() ) gHGCal = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
  if( producesHEfrontDigis() ) gHGCal = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
  if( producesHEbackDigis() )  gHcal = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));

  //accumulate in-time the main event
  if( nullptr != gHGCal ) {
    accumulate(hits, 0, gHGCal, hre);
  } else if( nullptr != gHcal ) {
    accumulate(hits, 0, gHcal, hre);
  } else {
    throw cms::Exception("BadConfiguration")
      << "HGCDigitizer is not producing EE, FH, or BH digis!";
  }
}

//
void HGCDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* hre) {

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //get geometry
  edm::ESHandle<CaloGeometry> geom;
  eventSetup.get<CaloGeometryRecord>().get(geom);
  
  const HGCalGeometry* gHGCal = nullptr;
  const HcalGeometry* gHcal = nullptr;
  if( producesEEDigis() ) gHGCal = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
  if( producesHEfrontDigis() ) gHGCal = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
  if( producesHEbackDigis() )  gHcal = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  
  //accumulate for the simulated bunch crossing  
  if( nullptr != gHGCal ) {
    accumulate(hits, e.bunchCrossing(), gHGCal, hre);
  } else if ( nullptr != gHcal ) {
    accumulate(hits, e.bunchCrossing(), gHcal, hre);
  } else {
    throw cms::Exception("BadConfiguration")
      << "HGCDigitizer is not producing EE, FH, or BH digis!";
  }
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, 
			      int bxCrossing,
			      const HcalGeometry* geom,
                              CLHEP::HepRandomEngine* hre) {
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, 
			      int bxCrossing,
			      const HGCalGeometry* geom,
                              CLHEP::HepRandomEngine* hre) {
  if( nullptr == geom ) return;
  const HGCalTopology &topo=geom->topology();
  const HGCalDDDConstants &dddConst=topo.dddConstants();

  //base time samples for each DetId, initialized to 0
  HGCCellInfo baseData;
  baseData.hit_info[0].fill(0.); //accumulated energy
  baseData.hit_info[1].fill(0.); //time-of-flight
  baseData.size = 0.0;
  baseData.thickness = std::numeric_limits<int>::max();
  
  //configuration to apply for the computation of time-of-flight
  bool weightToAbyEnergy(false);
  float tdcOnset(0.f),keV2fC(0.f);
  switch( mySubDet_ ) {
  case ForwardSubdetector::HGCEE:
    weightToAbyEnergy = theHGCEEDigitizer_->toaModeByEnergy();
    tdcOnset          = theHGCEEDigitizer_->tdcOnset();
    keV2fC            = theHGCEEDigitizer_->keV2fC();
    break;
  case ForwardSubdetector::HGCHEF:
    weightToAbyEnergy = theHGCHEfrontDigitizer_->toaModeByEnergy();
    tdcOnset          = theHGCHEfrontDigitizer_->tdcOnset();
    keV2fC            = theHGCHEfrontDigitizer_->keV2fC();
    break;
  case ForwardSubdetector::HGCHEB:
    weightToAbyEnergy = theHGCHEbackDigitizer_->toaModeByEnergy();
    tdcOnset          = theHGCHEbackDigitizer_->tdcOnset();
    keV2fC            = theHGCHEbackDigitizer_->keV2fC();     
    break;
  default:
    break;
  }

  //create list of tuples (pos in container, RECO DetId, time) to be sorted first
  int nchits=(int)hits->size();  
  std::vector< HGCCaloHitTuple_t > hitRefs(nchits);
  for(int i=0; i<nchits; i++) {
    const auto& the_hit = hits->at(i);
    int layer, cell, sec, subsec, zp;
    uint32_t simId = the_hit.id();
    const bool isSqr = (dddConst.geomMode() == HGCalGeometryMode::Square);
    if (isSqr) {
      HGCalTestNumbering::unpackSquareIndex(simId, zp, layer, sec, subsec, cell);
    } else {
      int subdet;
      HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell); 
      mySubDet_ = (ForwardSubdetector)(subdet);
      //sec is wafer and subsec is celltyp
    }
    //skip this hit if after ganging it is not valid
    std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,sec,topo.detectorType());
    cell  = recoLayerCell.first;
    layer = recoLayerCell.second;    
    if (layer<0 || cell<0) {
      hitRefs[i]=std::make_tuple( i, 0, 0. );
      continue;
    }

    //assign the RECO DetId
    DetId id;
    if (dddConst.geomMode() == HGCalGeometryMode::Square) {
      id = (producesEEDigis() ?	
	    (uint32_t)HGCEEDetId(mySubDet_,zp,layer,sec,subsec,cell):
	    (uint32_t)HGCHEDetId(mySubDet_,zp,layer,sec,subsec,cell) );
    } else {
      id = HGCalDetId(mySubDet_,zp,layer,subsec,sec,cell);
    }

    if (verbosity_>0) {
      if (producesEEDigis())
	  edm::LogInfo("HGCDigitizer") <<" i/p " << std::hex << simId << std::dec << " o/p " << HGCEEDetId(id) << std::endl;
      else
	  edm::LogInfo("HGCDigitizer") << " i/p " << std::hex << simId << std::dec << " o/p " << HGCHEDetId(id) << std::endl;
    }

    hitRefs[i]=std::make_tuple( i, 
				id.rawId(), 
				(float)the_hit.time() );
  }
  std::sort(hitRefs.begin(),hitRefs.end(),this->orderByDetIdThenTime);
  
  //loop over sorted hits
  for(int i=0; i<nchits; ++i) {
    const int hitidx   = std::get<0>(hitRefs[i]);
    const uint32_t id  = std::get<1>(hitRefs[i]);
    if(id==0) continue; // to be ignored at RECO level

    const float toa    = std::get<2>(hitRefs[i]);
    const PCaloHit &hit=hits->at( hitidx );     
    const float charge = hit.energy()*1e6*keV2fC;
      
    //distance to the center of the detector
    const float dist2center( geom->getPosition(id).mag() );
      
    //hit time: [time()]=ns  [centerDist]=cm [refSpeed_]=cm/ns + delay by 1ns
    //accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const float tof = toa-dist2center/refSpeed_+tofDelay_ ;
    const int itime= std::floor( tof/bxTime_ ) + 9;
      
    //no need to add bx crossing - tof comes already corrected from the mixing module
    //itime += bxCrossing;
    //itime += 9;
      
    if(itime<0 || itime>14) continue; 
    
    //check if already existing (perhaps could remove this in the future - 2nd event should have all defined)
    HGCSimHitDataAccumulator::iterator simHitIt=simHitAccumulator_->find(id);
    if(simHitIt == simHitAccumulator_->end()) {
      simHitIt = simHitAccumulator_->insert( std::make_pair(id,baseData) ).first;      
    }
      
    //check if time index is ok and store energy
    if(itime >= (int)simHitIt->second.hit_info[0].size() ) continue;

    (simHitIt->second).hit_info[0][itime] += charge;
    float accCharge=(simHitIt->second).hit_info[0][itime];
      
    //time-of-arrival (check how to be used)
      if(weightToAbyEnergy) (simHitIt->second).hit_info[1][itime] += charge*tof;
      else if((simHitIt->second).hit_info[1][itime]==0)
	{	
	  if( accCharge>tdcOnset)
	    {
	      //extrapolate linear using previous simhit if it concerns to the same DetId
	      float fireTDC=tof;
	      if(i>0)
		{
		  uint32_t prev_id  = std::get<1>(hitRefs[i-1]);
		  if(prev_id==id)
		    {
		      float prev_toa    = std::get<2>(hitRefs[i-1]);
		      float prev_tof(prev_toa-dist2center/refSpeed_+tofDelay_);
		      //float prev_charge = std::get<3>(hitRefs[i-1]);
		      float deltaQ2TDCOnset = tdcOnset-((simHitIt->second).hit_info[0][itime]-charge);
		      float deltaQ          = charge;
		      float deltaT          = (tof-prev_tof);
		      fireTDC               = deltaT*(deltaQ2TDCOnset/deltaQ)+prev_tof;
		    }		  
		}
	      
	      (simHitIt->second).hit_info[1][itime]=fireTDC;
	    }
	}
    }
  hitRefs.clear();
  
  //add base data for noise simulation (and thicknesses)
  if(!checkValidDetIds_) return;
  if(nullptr == geom) return;
  const std::vector<DetId> &validIds=geom->getValidDetIds();   
  int nadded(0);
  if (useAllChannels_) {
    for(std::vector<DetId>::const_iterator it=validIds.begin(); it!=validIds.end(); it++) {
      uint32_t id(it->rawId());
      auto itr = simHitAccumulator_->emplace(id, baseData);
      int waferTypeL = 0;
      bool isHalf = false;
      if(dddConst.geomMode() == HGCalGeometryMode::Square) {
        waferTypeL = producesEEDigis() ? 2 : 3;
        isHalf = false;
      } else {
        HGCalDetId hid(id);
        int wafer = HGCalDetId(id).wafer();
        waferTypeL = dddConst.waferTypeL(wafer);        
        isHalf = dddConst.isHalfCell(wafer,hid.cell());
      }
      itr.first->second.size = (isHalf ? 0.5 : 1.0);
      itr.first->second.thickness = waferTypeL;
      nadded++;
    }
  }
  
  if (verbosity_ > 0) 
    edm::LogInfo("HGCDigitizer") 
      << "Added " << nadded << ":" << validIds.size() 
      << " detIds without " << hitCollection_ 
      << " in first event processed" << std::endl;
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
  for( HGCSimHitDataAccumulator::iterator it = simHitAccumulator_->begin(); it!=simHitAccumulator_->end(); it++)
    {
      it->second.hit_info[0].fill(0.);
      it->second.hit_info[1].fill(0.);
    }
}





