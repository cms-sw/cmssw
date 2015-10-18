#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"

//
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps)
{
  edm::ParameterSet cfg = ps.getParameter<edm::ParameterSet>("digiCfg");
  keV2MIP_   = cfg.getParameter<double>("keV2MIP");
  noise_MIP_ = cfg.getParameter<double>("noise_MIP");
  nPEperMIP_ = cfg.getParameter<double>("nPEperMIP");
  nTotalPE_  = cfg.getParameter<double>("nTotalPE");
  xTalk_     = cfg.getParameter<double>("xTalk");
  sdPixels_  = cfg.getParameter<double>("sdPixels");
}

//
void HGCHEbackDigitizer::runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine)
{
  runCaliceLikeDigitizer(digiColl,simData,engine);
}
  
//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine)
{
 
  //switch to true if you want to print some details
  const bool debug(false);
  
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end();
      it++)
    {
      std::vector<float> chargeColl(it->second[0].size(),0);
      for(size_t i=0; i<it->second[0].size(); i++)
	{
	  //convert total energy GeV->keV->MIP
	  float totalIniMIPs( (it->second)[0][i]*1e6*keV2MIP_ );
	  	  
	  //generate random number of photon electrons
	  uint32_t npe = (uint32_t)CLHEP::RandPoisson::shoot(engine,totalIniMIPs*nPEperMIP_);
	  
	  //number of pixels	
	  float x=exp(-(float)(npe)/(float)(nTotalPE_));
	  uint32_t nPixel(0);
	  if(xTalk_*x!=1) nPixel=(uint32_t) std::max( float(nTotalPE_*(1-x)/(1-xTalk_*x)), float(0.) );
	  
	  //update signal
	  nPixel=(uint32_t)std::max( float(CLHEP::RandGauss::shoot(engine,(float)nPixel,(float)sdPixels_)), float(0.) );
	  
	  //convert to MIP again and saturate
	  float totalMIPs(totalIniMIPs);
	  if(nTotalPE_!=nPixel && (nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel)>0 )
	    totalMIPs = (nTotalPE_/nPEperMIP_)*log((nTotalPE_-xTalk_*nPixel)/(nTotalPE_-nPixel));
	  else
	    totalMIPs = 0;
	  
	  //add noise (in MIPs)
	  chargeColl[i]=totalMIPs+std::max(CLHEP::RandGauss::shoot(engine,0.,noise_MIP_),0.);
	  if(debug && (it->second)[0][i]>0) 
	    std::cout << "[runCaliceLikeDigitizer] En=" << (it->second)[0][i]*1e6 << " keV = " << totalIniMIPs << " MIPs -> " << chargeColl[i] << " MIPs" << std::endl;
	}	
      
      //init a new data frame and run shaper
      HGCHEDataFrame newDataFrame( it->first );
      myFEelectronics_->runTrivialShaper(newDataFrame,chargeColl);

      //prepare the output
      updateOutput(digiColl,newDataFrame);
    } 
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer()
{
}

