#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

//
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps) {
  try{
    edm::ParameterSet caliceSpec =  ps.getParameter<edm::ParameterSet>("digiCfg").getParameter<edm::ParameterSet>("caliceSpecific");
    nPEperMIP_ = caliceSpec.getParameter<double>("nPEperMIP");
    nTotalPE_  = caliceSpec.getParameter<double>("nTotalPE");
    xTalk_     = caliceSpec.getParameter<double>("xTalk");
    sdPixels_  = caliceSpec.getParameter<double>("sdPixels");
  }catch(std::exception &e){
    //no need to propagate
  }
}

//
void HGCHEbackDigitizer::runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
  switch(digitizationType) {
  case 1: 
    {
      runCaliceLikeDigitizer(digiColl,simData,engine);
      break;
    }
  }
}
  
//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine) {
 
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end(); it++) {
    //init a new data frame
    HGCHEDataFrame newDataFrame( it->first );

    for(size_t i=0; i<it->second.size(); i++) {
      //convert total energy GeV->keV->ADC counts
      float totalEn( (it->second)[i]*1e6 );
	  
      //convert energy to MIP
      float totalIniMIPs = totalEn/mipInKeV_;
	  
      //generate random number of photon electrons
      CLHEP::RandPoissonQ randPoissonQ(*engine, totalIniMIPs*nPEperMIP_);
      uint32_t npe = (uint32_t)randPoissonQ.fire();
	  
      //number of pixels	
      float x=exp(-(float)(npe)/(float)(nTotalPE_));
      uint32_t nPixel(0);
      if(xTalk_*x!=1) nPixel=(uint32_t) std::max( float(nTotalPE_*(1-x)/(1-xTalk_*x)), float(0.) );
	  
      //update signal
      nPixel=(uint32_t)std::max( float(CLHEP::RandGaussQ::shoot(engine,(float)nPixel,(float)sdPixels_)),float(0.) );
	  
      //convert to MIP again and saturate
      float totalMIPs(totalIniMIPs);
      if(nTotalPE_!=nPixel && (nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel)>0 )
	totalMIPs = (nTotalPE_/nPEperMIP_)*log((nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel));
      else
	totalMIPs = 0;
	  
      //add noise (in MIPs)
      double noiseMIPs=CLHEP::RandGaussQ::shoot(engine,0.,1./mip2noise_);
      totalMIPs=std::max(float(totalMIPs+noiseMIPs),float(0.));
	  
      //round to integer (sample will saturate the value according to available bits)
      uint16_t totalEnInt = floor( totalMIPs / lsbInMIP_ );
	 	  
      //0 gain for the moment
      HGCSample singleSample;
      singleSample.set(0, totalEnInt );
      newDataFrame.setSample(i, singleSample);

    }	
      
    //run shaper
    runShaper(newDataFrame);

    //prepare the output
    updateOutput(digiColl,newDataFrame);
  } 
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer() { }

