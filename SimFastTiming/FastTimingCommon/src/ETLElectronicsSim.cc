#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandGaussQ.h"

using namespace mtd;

ETLElectronicsSim::ETLElectronicsSim(const edm::ParameterSet& pset) :
  debug_( pset.getUntrackedParameter<bool>("debug",false) ),
  bxTime_(pset.getParameter<double>("bxTime") ),
  sigmaEta_( pset.getParameter<std::string>("etaResolution") ),
  adcNbits_( pset.getParameter<uint32_t>("adcNbits") ),
  tdcNbits_( pset.getParameter<uint32_t>("tdcNbits") ),
  adcSaturation_MIP_( pset.getParameter<double>("adcSaturation_MIP") ),
  adcLSB_MIP_( adcSaturation_MIP_/std::pow(2.,adcNbits_) ),
  adcThreshold_MIP_( pset.getParameter<double>("adcThreshold_MIP") ),
  toaLSB_ns_( pset.getParameter<double>("toaLSB_ns") ) {
}


void ETLElectronicsSim::run(const mtd::MTDSimHitDataAccumulator& input,
			    ETLDigiCollection& output,
			    CLHEP::HepRandomEngine *hre) const {
  
  MTDSimHitData chargeColl, toa;
  

  std::vector<double> emptyV;
  std::vector<double> eta(1);

  for(MTDSimHitDataAccumulator::const_iterator it=input.begin();
      it!=input.end();
      it++) {
    
    chargeColl.fill(0.f);
    toa.fill(0.f);
    for(size_t i=0; i<it->second.hit_info[0].size(); i++) {

      if ( (it->second).hit_info[0][i] < adcThreshold_MIP_ ) continue;

      // time of arrival
      float finalToA = (it->second).hit_info[1][i];

      // Gaussian smearing of the time of arrival
      eta[0] = 2.;  // This is just temporary. Once the RECO geometry is 
                    // available, the actual module eta will be used.
      double sigmaToA = sigmaEta_.evaluate(eta, emptyV);

      if ( sigmaToA > 0. )
	finalToA += CLHEP::RandGaussQ::shoot(hre, 0., sigmaToA);

      // fill the time and charge arrays
      const unsigned int ibucket = std::floor( finalToA/bxTime_ );
      if ( (i+ibucket) >= chargeColl.size() ) continue;
      
      chargeColl[i+ibucket] += (it->second).hit_info[0][i];      

      if ( toa[i+ibucket] == 0. || (finalToA-ibucket*bxTime_) < toa[i+ibucket] )
	toa[i+ibucket] = finalToA - ibucket*bxTime_;
      
    }

    // run the shaper to create a new data frame
    ETLDataFrame rawDataFrame( it->first.detid_ );    
    runTrivialShaper(rawDataFrame,chargeColl,toa, it->first.row_, it->first.column_);
    updateOutput(output,rawDataFrame);
    
  }
    
}

  
void ETLElectronicsSim::runTrivialShaper(ETLDataFrame &dataFrame, 
					 const mtd::MTDSimHitData& chargeColl,
					 const mtd::MTDSimHitData& toa,
					 const uint8_t row, const uint8_t col) const {
    bool debug = debug_;
#ifdef EDM_ML_DEBUG  
  for(int it=0; it<(int)(chargeColl.size()); it++) debug |= (chargeColl[it]>adcThreshold_fC_);
#endif
    
  if(debug) edm::LogVerbatim("ETLElectronicsSim") << "[runTrivialShaper]" << std::endl;
  
  //set new ADCs 
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {      
      //brute force saturation, maybe could to better with an exponential like saturation      
      const uint32_t adc=std::floor( std::min(chargeColl[it],adcSaturation_MIP_) / adcLSB_MIP_ );
      const uint32_t tdc_time=std::floor( toa[it] / toaLSB_ns_ );
      ETLSample newSample;
      newSample.set(chargeColl[it] > adcThreshold_MIP_,false,tdc_time,adc,row,col);
      dataFrame.setSample(it,newSample);

      if(debug) edm::LogVerbatim("ETLElectronicsSim") << adc << " (" << chargeColl[it] << "/" << adcLSB_MIP_ << ") ";
    }

  if(debug) { 
    std::ostringstream msg;
    dataFrame.print(msg);
    edm::LogVerbatim("ETLElectronicsSim") << msg.str() << std::endl;
  } 
}
  
void ETLElectronicsSim::updateOutput(ETLDigiCollection &coll,
				     const ETLDataFrame& rawDataFrame) const {
  int itIdx(9);
  if(rawDataFrame.size()<=itIdx+2) return;
  
  ETLDataFrame dataFrame( rawDataFrame.id() );
  dataFrame.resize(dfSIZE);
  bool putInEvent(false);
  for(int it=0;it<dfSIZE; ++it) {    
    dataFrame.setSample(it, rawDataFrame[itIdx-2+it]);
    if(it==2) putInEvent = rawDataFrame[itIdx-2+it].threshold(); 
  }

  if(putInEvent) {
    coll.push_back(dataFrame);    
  }
}
