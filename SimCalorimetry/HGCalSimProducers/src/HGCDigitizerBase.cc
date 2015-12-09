#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"

using namespace hgc_digi;

template<class DFr>
void HGCDigitizerBase<DFr>::run( std::auto_ptr<HGCDigitizerBase::DColl> &digiColl,
                                  HGCSimHitDataAccumulator &simData,
                                  uint32_t digitizationType,
                                  CLHEP::HepRandomEngine* engine) {
  if(digitizationType==0) runSimple(digiColl,simData,engine);
  else                    runDigitizer(digiColl,simData,digitizationType,engine);
}

template<class DFr>
void HGCDigitizerBase<DFr>::runSimple(std::auto_ptr<HGCDigitizerBase::DColl> &coll,
                                       HGCSimHitDataAccumulator &simData, 
                                       CLHEP::HepRandomEngine* engine) {
  HGCSimHitData chargeColl,toa;
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end();
      it++) {
    chargeColl.fill(0.f); 
    toa.fill(0.f);
    for(size_t i=0; i<it->second[0].size(); i++) {
      double rawCharge((it->second)[0][i]);
      
      //time of arrival
      toa[i]=(it->second)[1][i];
      if(myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE && rawCharge>0) 
        toa[i]=(it->second)[1][i]/rawCharge;
      
      //convert total energy in GeV to charge (fC)
      //double totalEn=rawEn*1e6*keV2fC_;
      double totalCharge=rawCharge;
      
      //add noise (in fC)
      //we assume it's randomly distributed and won't impact ToA measurement
      totalCharge += std::max( CLHEP::RandGaussQ::shoot(engine,0,noise_fC_) , 0. );
      if(totalCharge<0) totalCharge=0;
      
      chargeColl[i]= totalCharge;
    }
    
    //run the shaper to create a new data frame
    DFr rawDataFrame( it->first );
    myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, engine);
    
    //update the output according to the final shape
    updateOutput(coll,rawDataFrame);
  }  
}

template<class DFr>
void HGCDigitizerBase<DFr>::updateOutput(std::auto_ptr<HGCDigitizerBase::DColl> &coll,
                                          const DFr& rawDataFrame) {
  int itIdx(9);
  if(rawDataFrame.size()<=itIdx+2) return;
  
  DFr dataFrame( rawDataFrame.id() );
  dataFrame.resize(5);
  bool putInEvent(false);
  HGCSample singleSample;
  for(int it=0;it<5; it++) {    
    singleSample.set(rawDataFrame[itIdx-2+it].threshold(),
                     rawDataFrame[itIdx-2+it].mode(),
                     rawDataFrame[itIdx-2+it].toa(),
                     rawDataFrame[itIdx-2+it].data());
    dataFrame.setSample(it, singleSample);
    if(it==2) { putInEvent=rawDataFrame[itIdx-2+it].threshold(); }
  }
  if(putInEvent) coll->push_back(dataFrame);
}

// cause the compiler to generate the appropriate code
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
template class HGCDigitizerBase<HGCEEDataFrame>;
template class HGCDigitizerBase<HGCHEDataFrame>;
