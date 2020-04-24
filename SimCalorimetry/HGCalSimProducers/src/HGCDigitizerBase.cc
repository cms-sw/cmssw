#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

using namespace hgc_digi;

namespace {
  void addCellMetadata(HGCCellInfo& info, 
		       const HcalGeometry* geom,
		       const DetId& detid ) {
    //base time samples for each DetId, initialized to 0
    info.size = 1.0;
    info.thickness = 1.0;
  }
  
  void addCellMetadata(HGCCellInfo& info, 
		       const HGCalGeometry* geom, 
		       const DetId& detid ) {
    const auto& topo     = geom->topology();
    const auto& dddConst = topo.dddConstants();
    uint32_t id(detid.rawId());
    int waferTypeL = 0;
    bool isHalf = false;
    HGCalDetId hid(id);
    int wafer = HGCalDetId(id).wafer();
    waferTypeL = dddConst.waferTypeL(wafer);        
    isHalf = dddConst.isHalfCell(wafer,hid.cell());
    //base time samples for each DetId, initialized to 0
    info.size = (isHalf ? 0.5 : 1.0);
    info.thickness = waferTypeL;
  }
  
  void addCellMetadata(HGCCellInfo& info,
		       const CaloSubdetectorGeometry* geom,
		       const DetId& detid ) {
    if( DetId::Hcal == detid.det() ) {
      const HcalGeometry* hc = static_cast<const HcalGeometry*>(geom);
      addCellMetadata(info,hc,detid);
    } else {
      const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
      addCellMetadata(info,hg,detid);
    }
  }
  
}


template<class DFr>
HGCDigitizerBase<DFr>::HGCDigitizerBase(const edm::ParameterSet& ps) {
  bxTime_        = ps.getParameter<double>("bxTime");
  myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg");
  doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
  if(myCfg_.exists("keV2fC"))   keV2fC_   = myCfg_.getParameter<double>("keV2fC");
  else                          keV2fC_   = 1.0;

  if( myCfg_.existsAs<std::vector<double> >( "chargeCollectionEfficiencies" ) ) {
    cce_ = myCfg_.getParameter<std::vector<double> >("chargeCollectionEfficiencies");
  } else {
    std::vector<double>().swap(cce_);
  }

  if(myCfg_.existsAs<double>("noise_fC")) {
    noise_fC_.resize(1);
    noise_fC_[0] = myCfg_.getParameter<double>("noise_fC");
  } else if ( myCfg_.existsAs<std::vector<double> >("noise_fC") ) {
    const auto& noises = myCfg_.getParameter<std::vector<double> >("noise_fC");
    noise_fC_.resize(0);
    noise_fC_.reserve(noises.size());
    for( auto noise : noises ) { noise_fC_.push_back( noise ); }
  } else {
    noise_fC_.resize(1);
    noise_fC_[0] = 1.f;
  }
  edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
  myFEelectronics_        = std::unique_ptr<HGCFEElectronics<DFr> >( new HGCFEElectronics<DFr>(feCfg) );
  myFEelectronics_->SetNoiseValues(noise_fC_); 
}

template<class DFr>
void HGCDigitizerBase<DFr>::run( std::unique_ptr<HGCDigitizerBase::DColl> &digiColl,
				 HGCSimHitDataAccumulator &simData,
				 const CaloSubdetectorGeometry* theGeom, 
				 const std::unordered_set<DetId>& validIds,
				 uint32_t digitizationType,
				 CLHEP::HepRandomEngine* engine) {
  if(digitizationType==0) runSimple(digiColl,simData,theGeom,validIds,engine);
  else                    runDigitizer(digiColl,simData,theGeom,validIds,digitizationType,engine);
}

template<class DFr>
void HGCDigitizerBase<DFr>::runSimple(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
				      HGCSimHitDataAccumulator &simData, 
				      const CaloSubdetectorGeometry* theGeom, 
				      const std::unordered_set<DetId>& validIds,
				      CLHEP::HepRandomEngine* engine) {
  HGCSimHitData chargeColl,toa;

  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  for( const auto& id : validIds ) {
    chargeColl.fill(0.f); 
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);    
    HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
    addCellMetadata(cell,theGeom,id);
    
    for(size_t i=0; i<cell.hit_info[0].size(); i++) {
      double rawCharge(cell.hit_info[0][i]);
      
      //time of arrival
      toa[i]=cell.hit_info[1][i];
      if(myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE && rawCharge>0) 
        toa[i]=cell.hit_info[1][i]/rawCharge;
      
      //convert total energy in GeV to charge (fC)
      //double totalEn=rawEn*1e6*keV2fC_;
      float totalCharge=rawCharge;
      
      //add noise (in fC)
      //we assume it's randomly distributed and won't impact ToA measurement
      //also assume that it is related to the charge path only and that noise fluctuation for ToA circuit be handled separately
      totalCharge += std::max( (float)CLHEP::RandGaussQ::shoot(engine,0.0,cell.size*noise_fC_[cell.thickness-1]) , 0.f );
      if(totalCharge<0.f) totalCharge=0.f;
      
      chargeColl[i]= totalCharge;
    }
    
    //run the shaper to create a new data frame
    DFr rawDataFrame( id );
    if( !cce_.empty() )
      myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, cell.thickness, engine, cce_[cell.thickness-1]);
    else
      myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, cell.thickness, engine);

    //update the output according to the final shape
    updateOutput(coll,rawDataFrame);
  }   
}

template<class DFr>
void HGCDigitizerBase<DFr>::updateOutput(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
                                          const DFr& rawDataFrame) {
  int itIdx(9);
  if(rawDataFrame.size()<=itIdx+2) return;
  
  DFr dataFrame( rawDataFrame.id() );
  dataFrame.resize(5);
  bool putInEvent(false);
  for(int it=0;it<5; it++) {    
    dataFrame.setSample(it, rawDataFrame[itIdx-2+it]);
    if(it==2) putInEvent = rawDataFrame[itIdx-2+it].threshold(); 
  }

  if(putInEvent) {
    coll->push_back(dataFrame);    
  }
}

// cause the compiler to generate the appropriate code
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
template class HGCDigitizerBase<HGCEEDataFrame>;
template class HGCDigitizerBase<HGCBHDataFrame>;
