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
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

#include <algorithm>
#include <boost/foreach.hpp>
#include "FWCore/Utilities/interface/transform.h"

using namespace hgc_digi;

namespace {
  
  constexpr std::array<double,3> occupancyGuesses = { { 0.5,0.2,0.2 } };


  float getPositionDistance(const HGCalGeometry* geom, const DetId& id) {
    return geom->getPosition(id).mag();
  }

  float getPositionDistance(const HcalGeometry* geom, const DetId& id) {
    return geom->getGeometry(id)->getPosition().mag();
  }

  int getCellThickness(const HGCalGeometry* geom, const DetId& detid ) {
    const auto& topo     = geom->topology();
    const auto& dddConst = topo.dddConstants();
    uint32_t id(detid.rawId());
    HGCalDetId hid(id);
    int wafer = HGCalDetId(id).wafer();
    int waferTypeL = dddConst.waferTypeL(wafer);
    return waferTypeL;
  }

  int getCellThickness(const HcalGeometry* geom, const DetId& detid ) {
    return 1;
  }

  void getValidDetIds(const HGCalGeometry* geom, std::unordered_set<DetId>& valid) {
    const std::vector<DetId>& ids = geom->getValidDetIds();
    valid.reserve(ids.size());
    valid.insert(ids.begin(),ids.end());
  }

  void getValidDetIds(const HcalGeometry* geom, std::unordered_set<DetId>& valid) {
    const std::vector<DetId>& ids = geom->getValidDetIds();
    for( const auto& id : ids ) {
      if( HcalEndcap == id.subdetId() &&
	  DetId::Hcal == id.det() ) 
	valid.emplace(id);
    }
    valid.reserve(valid.size());
  }

  DetId simToReco(const HcalGeometry* geom, unsigned simid) {
    DetId result(0);
    const auto& topo     = geom->topology();
    const auto* dddConst = topo.dddConstants();
    HcalDetId id = HcalHitRelabeller::relabel(simid,dddConst);

    if (id.subdet()==int(HcalEndcap)) {
      result = id;    
    }

    return result;
  }

  DetId simToReco(const HGCalGeometry* geom, unsigned simId) {
    DetId result(0);
    const auto& topo     = geom->topology();
    const auto& dddConst = topo.dddConstants();
    
    int subdet(DetId(simId).subdetId()), layer, cell, sec, subsec, zp;

    const bool isSqr = (dddConst.geomMode() == HGCalGeometryMode::Square);
    if (isSqr) {
      HGCalTestNumbering::unpackSquareIndex(simId, zp, layer, sec, subsec, cell);
    } else {
      HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell); 
      //sec is wafer and subsec is celltyp
    }
    //skip this hit if after ganging it is not valid
    std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,sec,topo.detectorType());
    cell  = recoLayerCell.first;
    layer = recoLayerCell.second;    
    if (layer<0 || cell<0) {
      return result;
    }

    //assign the RECO DetId
    result = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
    
    return result;
  }  

  float getCCE(const HGCalGeometry* geom,
	       const DetId& detid,
	       const std::vector<float>&cces) {
    if( cces.empty() ) return 1.f;
    const auto& topo     = geom->topology();
    const auto& dddConst = topo.dddConstants();
    uint32_t id(detid.rawId());
    HGCalDetId hid(id);
    int wafer = HGCalDetId(id).wafer();
    int waferTypeL = dddConst.waferTypeL(wafer);  
    return cces[waferTypeL-1];
  }

  float getCCE(const HcalGeometry* geom,
	       const DetId& id,
	       const std::vector<float>&cces) {
    return 1.f;
  }

}

//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps,
                           edm::ConsumesCollector& iC) :
  simHitAccumulator_( new HGCSimHitDataAccumulator() ),
  mySubDet_(ForwardSubdetector::ForwardEmpty),
  refSpeed_(0.1*CLHEP::c_light), //[CLHEP::c_light]=mm/ns convert to cm/ns
  averageOccupancies_(occupancyGuesses),
  nEvents_(1)
{
  //configure from cfg
  hitCollection_     = ps.getParameter< std::string >("hitCollection");
  digiCollection_    = ps.getParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getParameter< double >("bxTime");
  digitizationType_  = ps.getParameter< uint32_t >("digitizationType");
  verbosity_         = ps.getUntrackedParameter< uint32_t >("verbosity",0);
  tofDelay_          = ps.getParameter< double >("tofDelay");  

  std::unordered_set<DetId>().swap(validIds_);
  
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag("g4SimHits",hitCollection_));
  const auto& myCfg_ = ps.getParameter<edm::ParameterSet>("digiCfg");
  
  if( myCfg_.existsAs<std::vector<double> >( "chargeCollectionEfficiencies" ) ) {
    cce_.clear();
    const auto& temp = myCfg_.getParameter<std::vector<double> >("chargeCollectionEfficiencies");
    for( double cce : temp ) {
      cce_.push_back(cce);
    }
  } else {
    std::vector<float>().swap(cce_);
  }
  
  if(hitCollection_.find("HitsEE")!=std::string::npos) { 
    mySubDet_=ForwardSubdetector::HGCEE;  
    theHGCEEDigitizer_=std::unique_ptr<HGCEEDigitizer>(new HGCEEDigitizer(ps) ); 
  }
  if(hitCollection_.find("HitsHEfront")!=std::string::npos)  
    { 
      mySubDet_=ForwardSubdetector::HGCHEF;
      theHGCHEfrontDigitizer_=std::unique_ptr<HGCHEfrontDigitizer>(new HGCHEfrontDigitizer(ps) );
    }
  if(hitCollection_.find("HcalHits")!=std::string::npos)
    { 
      mySubDet_=ForwardSubdetector::HGCHEB;
      theHGCHEbackDigitizer_=std::unique_ptr<HGCHEbackDigitizer>(new HGCHEbackDigitizer(ps) );
    }
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es) 
{
  // reserve memory for a full detector
  unsigned idx = std::numeric_limits<unsigned>::max();
  switch(mySubDet_) {
  case ForwardSubdetector::HGCEE:
    idx = 0;
    break;
  case ForwardSubdetector::HGCHEF:
    idx = 1;
    break;
  case ForwardSubdetector::HGCHEB:
    idx = 2;
    break;
  default:
    break;
  }
  simHitAccumulator_->reserve( averageOccupancies_[idx]*validIds_.size() );
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es, CLHEP::HepRandomEngine* hre)
{
  hitRefs_bx0.clear();
  
  const CaloSubdetectorGeometry* theGeom = ( nullptr == gHGCal_ ? 
					     static_cast<const CaloSubdetectorGeometry*>(gHcal_) : 
					     static_cast<const CaloSubdetectorGeometry*>(gHGCal_)  );
  
  ++nEvents_;
  unsigned idx = std::numeric_limits<unsigned>::max();
  switch(mySubDet_) {
  case ForwardSubdetector::HGCEE:
    idx = 0;
    break;
  case ForwardSubdetector::HGCHEF:
    idx = 1;
    break;
  case ForwardSubdetector::HGCHEB:
    idx = 2;
    break;
  default:
    break;
  }
  // release memory for unfilled parts of hash table
  if( validIds_.size()*averageOccupancies_[idx] > simHitAccumulator_->size() ) {
    simHitAccumulator_->reserve(simHitAccumulator_->size());
  }
  //update occupancy guess
  const double thisOcc = simHitAccumulator_->size()/((double)validIds_.size());
  averageOccupancies_[idx] = (averageOccupancies_[idx]*(nEvents_-1) + thisOcc)/nEvents_;
  
  if( producesEEDigis() ) 
    {
      std::unique_ptr<HGCEEDigiCollection> digiResult(new HGCEEDigiCollection() );
      theHGCEEDigitizer_->run(digiResult,*simHitAccumulator_,theGeom,validIds_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";      
      e.put(std::move(digiResult),digiCollection());
    }
  if( producesHEfrontDigis())
    {
      std::unique_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEfrontDigitizer_->run(digiResult,*simHitAccumulator_,theGeom,validIds_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
      e.put(std::move(digiResult),digiCollection());
    }
  if( producesHEbackDigis() )
    {
      std::unique_ptr<HGCBHDigiCollection> digiResult(new HGCBHDigiCollection() );
      theHGCHEbackDigitizer_->run(digiResult,*simHitAccumulator_,theGeom,validIds_,digitizationType_, hre);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE back hits";
      e.put(std::move(digiResult),digiCollection());
    }
  
  hgc::HGCSimHitDataAccumulator().swap(*simHitAccumulator_);
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
  
  //accumulate in-time the main event
  if( nullptr != gHGCal_ ) {
    accumulate(hits, 0, gHGCal_, hre);
  } else if( nullptr != gHcal_ ) {
    accumulate(hits, 0, gHcal_, hre);
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
  
  //accumulate for the simulated bunch crossing  
  if( nullptr != gHGCal_ ) {
    accumulate(hits, e.bunchCrossing(), gHGCal_, hre);
  } else if ( nullptr != gHcal_ ) {
    accumulate(hits, e.bunchCrossing(), gHcal_, hre);
  } else {
    throw cms::Exception("BadConfiguration")
      << "HGCDigitizer is not producing EE, FH, or BH digis!";
  }
}

//
template<typename GEOM>
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, 
			      int bxCrossing,
			      const GEOM* geom,
                              CLHEP::HepRandomEngine* hre) {
  if( nullptr == geom ) return;
  
  
  
  //configuration to apply for the computation of time-of-flight
  bool weightToAbyEnergy(false);
  std::array<float, 3> tdcForToAOnset{ {0.f, 0.f, 0.f} };
  float keV2fC(0.f);
  switch( mySubDet_ ) {
  case ForwardSubdetector::HGCEE:
    weightToAbyEnergy = theHGCEEDigitizer_->toaModeByEnergy();
    tdcForToAOnset    = theHGCEEDigitizer_->tdcForToAOnset();
    keV2fC            = theHGCEEDigitizer_->keV2fC();
    break;
  case ForwardSubdetector::HGCHEF:
    weightToAbyEnergy = theHGCHEfrontDigitizer_->toaModeByEnergy();
    tdcForToAOnset    = theHGCHEfrontDigitizer_->tdcForToAOnset();
    keV2fC            = theHGCHEfrontDigitizer_->keV2fC();
    break;
  case ForwardSubdetector::HGCHEB:
    weightToAbyEnergy = theHGCHEbackDigitizer_->toaModeByEnergy();
    tdcForToAOnset    = theHGCHEbackDigitizer_->tdcForToAOnset();
    keV2fC            = theHGCHEbackDigitizer_->keV2fC();     
    break;
  default:
    break;
  }

  //create list of tuples (pos in container, RECO DetId, time) to be sorted first
  int nchits=(int)hits->size();  
  std::vector< HGCCaloHitTuple_t > hitRefs;
  hitRefs.reserve(nchits);
  for(int i=0; i<nchits; ++i) {
    const auto& the_hit = hits->at(i);    
    
    DetId id = simToReco(geom,the_hit.id());
    
    if (verbosity_>0) {
      if (producesEEDigis())
	edm::LogInfo("HGCDigitizer") << " i/p " << std::hex << the_hit.id() << " o/p " << id.rawId() << std::dec << std::endl;
      else
	edm::LogInfo("HGCDigitizer") << " i/p " << std::hex << the_hit.id() << " o/p " << id.rawId() << std::dec << std::endl;
    }

    if( 0 != id.rawId() ) {      
      hitRefs.emplace_back( i, id.rawId(), (float)the_hit.time() );
    }
  }
  std::sort(hitRefs.begin(),hitRefs.end(),this->orderByDetIdThenTime);
  
  //loop over sorted hits
  nchits = hitRefs.size();
  for(int i=0; i<nchits; ++i) {
    const int hitidx   = std::get<0>(hitRefs[i]);
    const uint32_t id  = std::get<1>(hitRefs[i]);

    //get the data for this cell, if not available then we skip it
   
    if( !validIds_.count(id) ) continue;
    HGCSimHitDataAccumulator::iterator simHitIt = simHitAccumulator_->emplace(id,HGCCellInfo()).first;

    if(id==0) continue; // to be ignored at RECO level

    const float toa    = std::get<2>(hitRefs[i]);
    const PCaloHit &hit=hits->at( hitidx );     
    const float charge = hit.energy()*1e6*keV2fC*getCCE(geom,id,cce_);
    
    //distance to the center of the detector
    const float dist2center( getPositionDistance(geom,id) );
      
    //hit time: [time()]=ns  [centerDist]=cm [refSpeed_]=cm/ns + delay by 1ns
    //accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const float tof = toa-dist2center/refSpeed_+tofDelay_ ;
    const int itime= std::floor( tof/bxTime_ ) + 9;

    //no need to add bx crossing - tof comes already corrected from the mixing module
    //itime += bxCrossing;
    //itime += 9;
      
    if(itime<0 || itime>14) continue;     
          
    //check if time index is ok and store energy
    if(itime >= (int)simHitIt->second.hit_info[0].size() ) continue;

    (simHitIt->second).hit_info[0][itime] += charge;


    //working version with pileup only for in-time hits
    int waferThickness = getCellThickness(geom,id);
    bool orderChanged = false;
    if(itime == 9){
      if(hitRefs_bx0[id].empty()){
	hitRefs_bx0[id].push_back(std::pair<float, float>(charge, tof));
      }
      else if(tof <= hitRefs_bx0[id].back().second){
	std::vector<std::pair<float, float> >::iterator findPos = 
	  std::upper_bound(hitRefs_bx0[id].begin(), hitRefs_bx0[id].end(), std::pair<float, float>(0.f,tof), 
			   [](const auto& i, const auto& j){return i.second < j.second;});

	std::vector<std::pair<float, float> >::iterator insertedPos = 
	  hitRefs_bx0[id].insert(findPos, (findPos == hitRefs_bx0[id].begin()) ? 
				 std::pair<float, float>(charge,tof) : std::pair<float, float>((findPos-1)->first+charge,tof));

	for(std::vector<std::pair<float, float> >::iterator step = insertedPos+1; step != hitRefs_bx0[id].end(); ++step){
	  step->first += charge;
	  if(step->first > tdcForToAOnset[waferThickness-1] && step->second != hitRefs_bx0[id].back().second){
	    hitRefs_bx0[id].resize(std::upper_bound(hitRefs_bx0[id].begin(), hitRefs_bx0[id].end(), std::pair<float, float>(0.f,step->second),
						    [](const auto& i, const auto& j){return i.second < j.second;}) - hitRefs_bx0[id].begin());
	    for(auto stepEnd = step+1; stepEnd != hitRefs_bx0[id].end(); ++stepEnd) stepEnd->first += charge;
	    break;
	  }
	}
	orderChanged = true;
      }
      else{
        if(hitRefs_bx0[id].back().first <= tdcForToAOnset[waferThickness-1]){
          hitRefs_bx0[id].push_back(std::pair<float, float>(hitRefs_bx0[id].back().first+charge, tof));
        }
      }
    }

    float accChargeForToA = hitRefs_bx0[id].empty() ? 0.f : hitRefs_bx0[id].back().first;

    //time-of-arrival (check how to be used)
    if(weightToAbyEnergy) (simHitIt->second).hit_info[1][itime] += charge*tof;
    else if(accChargeForToA > tdcForToAOnset[waferThickness-1] &&
	    ((simHitIt->second).hit_info[1][itime] == 0 || orderChanged == true) ){
      float fireTDC = hitRefs_bx0[id].back().second;
      if (hitRefs_bx0[id].size() > 1){
	float chargeBeforeThr = 0.f;
	float tofchargeBeforeThr = 0.f;
	for(const auto& step : hitRefs_bx0[id]){
	  if(step.first + chargeBeforeThr <= tdcForToAOnset[waferThickness-1]){
	    chargeBeforeThr += step.first;
	    tofchargeBeforeThr = step.second;
	  }
	  else break;
	}
	float deltaQ = accChargeForToA - chargeBeforeThr;
	float deltaTOF = fireTDC - tofchargeBeforeThr;
	fireTDC = (tdcForToAOnset[waferThickness-1] - chargeBeforeThr) * deltaTOF / deltaQ + tofchargeBeforeThr;
      }
      (simHitIt->second).hit_info[1][itime] = fireTDC;                                                                  
    }
    
  }
  hitRefs.clear();
}

//
void HGCDigitizer::beginRun(const edm::EventSetup & es)
{
  //get geometry
  edm::ESHandle<CaloGeometry> geom;
  es.get<CaloGeometryRecord>().get(geom);
  
  gHGCal_ = nullptr;
  gHcal_ = nullptr;

  if( producesEEDigis() )      gHGCal_ = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
  if( producesHEfrontDigis() ) gHGCal_ = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
  if( producesHEbackDigis() )  gHcal_  = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
  
  int nadded(0);  
  //valid ID lists
  if( nullptr != gHGCal_ ) {
    getValidDetIds( gHGCal_, validIds_ );    
  } else if( nullptr != gHcal_ ) {
    getValidDetIds( gHcal_, validIds_ );    
  } else {
    throw cms::Exception("BadConfiguration")
      << "HGCDigitizer is not producing EE, FH, or BH digis!";
  }

  if (verbosity_ > 0) 
    edm::LogInfo("HGCDigitizer") 
      << "Added " << nadded << ":" << validIds_.size() 
      << " detIds without " << hitCollection_ 
      << " in first event processed" << std::endl;
}

//
void HGCDigitizer::endRun()
{
  std::unordered_set<DetId>().swap(validIds_);
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

template void HGCDigitizer::accumulate<HcalGeometry>(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const HcalGeometry *geom, CLHEP::HepRandomEngine* hre);
template void HGCDigitizer::accumulate<HGCalGeometry>(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const HGCalGeometry *geom, CLHEP::HepRandomEngine* hre);
