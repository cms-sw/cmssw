#include "SimCalorimetry/HGCSimProducers/interface/HGCHEbackDigitizer.h"

//
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps)
{
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
void HGCHEbackDigitizer::setRandomNumberEngine(CLHEP::HepRandomEngine& engine) 
{
  peGen_ = new CLHEP::RandPoisson(engine,1.0);
  sigGen_= new CLHEP::RandGauss(engine,0.0,1.0);
  HGCDigitizerBase::setRandomNumberEngine(engine);
}

//
void HGCHEbackDigitizer::runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType)
{
  switch(digitizationType)
    {
    case 1: 
      {
	runCaliceLikeDigitizer(digiColl,simData);
	break;
      }
    }
}
  
//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData)
{
 
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end();
      it++)
    {
      std::vector<float> chargeColl(it->second[0].size(),0),toa(it->second[0].size(),0);
      for(size_t i=0; i<it->second[0].size(); i++)
	{
	  //convert total energy GeV->keV->ADC counts
	  float totalEn( (it->second)[0][i]*1e6 );
	  if(totalEn>0) toa[i]=(it->second)[1][i]*1e6/totalEn;
	  
	  //convert energy to MIP
	  float totalIniMIPs = totalEn/mipInKeV_;
	  
	  //generate random number of photon electrons
	  uint32_t npe = (uint32_t)peGen_->fire(totalIniMIPs*nPEperMIP_);
	  
	  //number of pixels	
	  float x=exp(-(float)(npe)/(float)(nTotalPE_));
	  uint32_t nPixel(0);
	  if(xTalk_*x!=1) nPixel=(uint32_t) std::max( float(nTotalPE_*(1-x)/(1-xTalk_*x)), float(0.) );
	  
	  //update signal
	  nPixel=(uint32_t)std::max( float(sigGen_->fire((float)nPixel,(float)sdPixels_)), float(0.) );
	  
	  //convert to MIP again and saturate
	  float totalMIPs(totalIniMIPs);
	  if(nTotalPE_!=nPixel && (nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel)>0 )
	    totalMIPs = (nTotalPE_/nPEperMIP_)*log((nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel));
	  else
	    totalMIPs = 0;
	  
	  //add noise (in MIPs)
	  double noiseMIPs=simpleNoiseGen_->fire(0.,1./mip2noise_);
	  totalMIPs=std::max(float(totalMIPs+noiseMIPs),float(0.));
	  chargeColl[i]=totalMIPs;
	}	
      
      //init a new data frame and run shaper
      HGCHEDataFrame newDataFrame( it->first );
      myFEelectronics_->runShaper(newDataFrame,chargeColl,toa);

      //prepare the output
      updateOutput(digiColl,newDataFrame);
    } 
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer()
{
}

