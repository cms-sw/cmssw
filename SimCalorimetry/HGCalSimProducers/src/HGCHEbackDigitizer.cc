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
void HGCHEbackDigitizer::runDigitizer(std::unique_ptr<HGCBHDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine)
{
  runCaliceLikeDigitizer(digiColl,simData,engine);
}
  
//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::unique_ptr<HGCBHDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine)
{
  //switch to true if you want to print some details
  constexpr bool debug(false);
  
  HGCSimHitData chargeColl;
  for(HGCSimHitDataAccumulator::iterator it=simData.begin();
      it!=simData.end();
      it++)
    {
      chargeColl.fill(0.f);      
      for(size_t i=0; i<it->second.hit_info[0].size(); ++i)
	{          
	  //convert total energy keV->MIP, since converted to keV in accumulator
	  const float totalIniMIPs( (it->second).hit_info[0][i]*keV2MIP_ );
          //std::cout << "energy in MIP: " << std::scientific << totalIniMIPs << std::endl;

	  //generate random number of photon electrons
	  const uint32_t npe = std::floor(CLHEP::RandPoissonQ::shoot(engine,totalIniMIPs*nPEperMIP_));
          
	  //number of pixels	
	  const float x = vdt::fast_expf( -((float)npe)/nTotalPE_ );
	  uint32_t nPixel(0);
	  if(xTalk_*x!=1) nPixel=(uint32_t) std::max( nTotalPE_*(1.f-x)/(1.f-xTalk_*x), 0.f );
	            
	  //update signal
	  nPixel = (uint32_t)std::max( CLHEP::RandGaussQ::shoot(engine,(double)nPixel,sdPixels_), 0. );
	            
	  //convert to MIP again and saturate
          float totalMIPs(0.f), xtalk = 0.f; 
          const float peDiff = nTotalPE_ - (float) nPixel;
          if (peDiff != 0.f) {
            xtalk = (nTotalPE_-xTalk_*((float)nPixel)) / peDiff;
            if( xtalk > 0.f && nPEperMIP_ != 0.f)
              totalMIPs = (nTotalPE_/nPEperMIP_)*vdt::fast_logf(xtalk);
          }
          
	  //add noise (in MIPs)
	  chargeColl[i] = totalMIPs+std::max( CLHEP::RandGaussQ::shoot(engine,0.,noise_MIP_), 0. );
	  if(debug && (it->second).hit_info[0][i]>0) 
	    std::cout << "[runCaliceLikeDigitizer] xtalk=" << xtalk << " En=" << (it->second).hit_info[0][i] << " keV -> " << totalIniMIPs << " raw-MIPs -> " << chargeColl[i] << " digi-MIPs" << std::endl;          
	}	
      
      //init a new data frame and run shaper
      HGCBHDataFrame newDataFrame( it->first );
      myFEelectronics_->runTrivialShaper( newDataFrame, chargeColl, 1 );

      //prepare the output
      updateOutput(digiColl,newDataFrame);
    } 
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer()
{
}

