#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "vdt/vdtMath.h"

using namespace hgc_digi;

//
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps)
{
  edm::ParameterSet cfg = ps.getParameter<edm::ParameterSet>("digiCfg");
  keV2MIP_   = cfg.getParameter<double>("keV2MIP");
  keV2fC_    = 1.0; //keV2MIP_; // hack for HEB
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
  constexpr bool debug(false);
  
  HGCSimHitData chargeColl;
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end();
      it++)
    {
      chargeColl.fill(0.f);      
      for(size_t i=0; i<it->second[0].size(); ++i)
	{          
	  //convert total energy keV->MIP, since converted to keV in accumulator
	  float totalIniMIPs( (it->second)[0][i]*keV2MIP_ );
          //std::cout << "energy in MIP: " << std::scientific << totalIniMIPs << std::endl;

	  //generate random number of photon electrons
	  uint32_t npe = std::floor(CLHEP::RandPoissonQ::shoot(engine,totalIniMIPs*nPEperMIP_));
          
	  //number of pixels	
	  float x = vdt::fast_expf( -((float)npe)/nTotalPE_ );
	  uint32_t nPixel(0);
	  if(xTalk_*x!=1) nPixel=(uint32_t) std::max( nTotalPE_*(1.f-x)/(1.f-xTalk_*x), 0.f );
	            
	  //update signal
	  nPixel = (uint32_t)std::max( CLHEP::RandGaussQ::shoot(engine,(double)nPixel,sdPixels_), 0. );
	            
	  //convert to MIP again and saturate
	  float totalMIPs(totalIniMIPs);
          const float xtalk = (nTotalPE_-xTalk_*((float)nPixel))/(nTotalPE_-((float)nPixel));
	  if( nTotalPE_ != nPixel && xtalk > 0. )
	    totalMIPs = (nTotalPE_/nPEperMIP_)*vdt::fast_logf(xtalk);
	  else
	    totalMIPs = 0.f;
	  
	  //add noise (in MIPs)
	  chargeColl[i] = totalMIPs+std::max( CLHEP::RandGaussQ::shoot(engine,0.,noise_MIP_), 0. );
	  if(debug && (it->second)[0][i]>0) 
	    std::cout << "[runCaliceLikeDigitizer] xtalk=" << xtalk << " En=" << (it->second)[0][i] << " keV -> " << totalIniMIPs << " raw-MIPs -> " << chargeColl[i] << " digi-MIPs" << std::endl;          
	}	
      
      //init a new data frame and run shaper
      HGCHEDataFrame newDataFrame( it->first );
      myFEelectronics_->runTrivialShaper( newDataFrame, chargeColl );

      //prepare the output
      updateOutput(digiColl,newDataFrame);
    } 
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer()
{
}

