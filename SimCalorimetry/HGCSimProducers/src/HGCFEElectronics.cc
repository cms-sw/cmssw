#include "SimCalorimetry/HGCSimProducers/interface/HGCFEElectronics.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

using namespace std;

//
template<class D>
HGCFEElectronics<D>::HGCFEElectronics(const edm::ParameterSet &ps)
{
  fwVersion_                      = ps.getParameter< uint32_t >("fwVersion");
  std::cout << "[HGCFEElectronics] running with version " << fwVersion_ << std::endl;
  if( ps.exists("adcPulse") )                       adcPulse_                       = ps.getParameter< std::vector<double> >("adcPulse");
  if( ps.exists("adcLSB_fC") )                      adcLSB_fC_                      = ps.getParameter<double>("adcLSB_fC");
  if( ps.exists("tdcLSB_fC") )                      tdcLSB_fC_                      = ps.getParameter<double>("tdcLSB_fC");
  if( ps.exists("adcThreshold_fC") )                adcThreshold_fC_                = ps.getParameter<double>("adcThreshold_fC");
  if( ps.exists("tdcOnset_fC") )                    tdcOnset_fC_                    = ps.getParameter<double>("tdcOnset_fC");
  if( ps.exists("tdcChargeDrainParameterisation") ) tdcChargeDrainParameterisation_ = ps.getParameter< std::vector<double> >("tdcChargeDrainParameterisation");
}


//
template<class D>
void HGCFEElectronics<D>::runTrivialShaper(D &dataFrame,std::vector<float> &chargeColl)
{
  bool debug(false);
  ////to enable debug uncomment me
  ////for(int it=0; it<(int)(chargeColl.size()); it++) debug |= (chargeColl[it]>adcThreshold_fC_*2);
    
  //set new ADCs
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      if(debug) cout << chargeColl[it] << " ";
      HGCSample newSample;
      newSample.set(chargeColl[it]>adcThreshold_fC_,false,0,floor(chargeColl[it]/adcLSB_fC_));
      dataFrame.setSample(it,newSample);
    }

  if(debug) dataFrame.print(std::cout);
}

//
template<class D>
void HGCFEElectronics<D>::runSimpleShaper(D &dataFrame,std::vector<float> &chargeColl)
{
  //convolute with pulse shape to compute new ADCs
  std::vector<float> newCharge(chargeColl.size(),0);
  bool debug(false);
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      float charge(chargeColl[it]);
      if(charge==0) continue;

      ////to enable debug uncomment me
      ////debug|=(charge>2*adcThreshold_fC_);
      if(debug) std::cout << "\t redistributing " << charge << " @ " << it;
      
      for(int ipulse=-2; ipulse<(int)(adcPulse_.size())-2; ipulse++)
	{
	  if(it+ipulse<0) continue;
	  if(it+ipulse>=(int)(dataFrame.size())) continue;
	  float chargeLeak=charge*adcPulse_[(ipulse+2)];
	  newCharge[it+ipulse]+= chargeLeak;
	  
	  if(debug) std::cout << " | " << it+ipulse << " " << chargeLeak;
	}
      
      if(debug) std::cout << std::endl;
    }

  //set new ADCs
  for(int it=0; it<(int)(newCharge.size()); it++)
    {
      if(debug) std::cout << newCharge[it] << " ";
      HGCSample newSample;
      newSample.set(newCharge[it]>adcThreshold_fC_,false,0,floor(newCharge[it]/adcLSB_fC_));
      dataFrame.setSample(it,newSample);      
    }
  
  if(debug) { std::cout << std::endl; dataFrame.print(std::cout); }
}

//
template<class D>
void HGCFEElectronics<D>::runShaperWithToT(D &dataFrame,std::vector<float> &chargeColl,std::vector<float> &toaColl)
{
  std::vector<bool>  busyFlags(chargeColl.size(),false);
  std::vector<float> newCharge(chargeColl.size(),0);
  std::vector<float> toaFromToT(chargeColl.size(),0);

  //first identify bunches which will trigger ToT
  bool debug(false);
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      float charge = chargeColl[it];
      float toa    = toaColl[it];
      if(charge < tdcOnset_fC_) 
	{
	  newCharge[it]=charge;
	  continue;
	}
      busyFlags[it]=true;

      ////to enable debug uncomment me
      debug=true;
      if(debug) std::cout << "Charge=" << charge << " with <toa>=" << toa << " ns, triggers ToT @ " << it << std::endl;

      //compute total charge to be integrated and integration time 
      //needs a loop as ToT will last as long as there is charge to dissipate
      int busyBxs(0);
      float totalCharge(charge), finalToA(toa), integTime(0);
      while(true)
	{
	  //compute integration time in ns and # bunches
	  float newIntegTime(0);
	  if(charge<tdcChargeDrainParameterisation_[0]) 
	    newIntegTime=tdcChargeDrainParameterisation_[1]*pow(totalCharge*1e-3,tdcChargeDrainParameterisation_[2])+tdcChargeDrainParameterisation_[3];
	  else                                          
	    newIntegTime=tdcChargeDrainParameterisation_[4]*pow(totalCharge*1e-3,tdcChargeDrainParameterisation_[5])+tdcChargeDrainParameterisation_[6];	  
	  int newBusyBxs=floor(newIntegTime/25.)+1;      

	  //if no update is needed regarding the number of bunches,
	  //then the ToT integration time has converged
	  integTime=newIntegTime;
	  if(newBusyBxs==busyBxs) break;

	  //update charge integrated during ToT
	  if(debug)
	    {
	      if(busyBxs==0) std::cout << "\t Intial busy estimate="<< integTime << " ns = " << newBusyBxs << " bxs" << std::endl;
	      else             std::cout << "\t ...integrated charge overflows initial busy estimate, interating again" << std::endl;
	    }

	  //update number of busy bunches
	  busyBxs=newBusyBxs;

	  //reset charge to be integrated
	  totalCharge=charge;
	  finalToA=toa*charge;

	  //add leakage from previous bunches in SARS ADC mode
	  for(int jt=0; jt<it; jt++)
	    {
	      if(busyFlags[jt]) continue;

	      float chargeDep_jt(chargeColl[jt]);
	      if(chargeDep_jt==0) continue;

	      int deltaT=(it-jt);
	      if(deltaT+2>(int)(adcPulse_.size())) continue;

	      float leakCharge( adcPulse_[deltaT+2]*chargeDep_jt );
	      if(debug) std::cout << "\t\t leaking " << chargeDep_jt << " fC @ deltaT=-" << deltaT << " -> +" << leakCharge << std::endl;

	      totalCharge  += leakCharge;
	      finalToA     += leakCharge*dataFrame[jt].toa(); 
	    }

	  //add contamination from posterior bunches
	  for(int jt=it+1; jt<it+busyBxs && jt<dataFrame.size() ; jt++) 
	    { 
	      //this charge will be integrated in TDC mode
	      //disable for SARS ADC
	      busyFlags[jt]=true; 

	      float extraCharge=chargeColl[jt];
	      if(extraCharge==0) continue;
	      if(debug) std::cout << "\t\t adding " << extraCharge << " fC @ deltaT=+" << (jt-it) << std::endl; 

	      totalCharge += extraCharge;
	      finalToA    += extraCharge*toaColl[jt];
	    }
	  
	  //finalize ToA contamination
	  finalToA /= totalCharge;
	}

      toaFromToT[it] = finalToA;
      newCharge[it]  = (totalCharge-tdcOnset_fC_);      

      if(debug) std::cout << "\t Final busy estimate="<< integTime << " ns = " << busyBxs << " bxs" << std::endl
			  << "\t Total integrated=" << totalCharge << " fC <toa>=" << toaFromToT[it] << " ns " << std::endl; 
      
      //last fC (tdcOnset) are dissipated trough pulse
      if(it+busyBxs<(int)(newCharge.size())) 
	{
	  float deltaT2nextBx((busyBxs*25-integTime));
	  float tdcOnsetLeakage(tdcOnset_fC_*exp(-deltaT2nextBx/tdcChargeDrainParameterisation_[7]));
	  if(debug) std::cout << "\t Leaking remainder of TDC onset " << tdcOnset_fC_ 
			      << " fC, to be dissipated in " << deltaT2nextBx 
			      << " ns, adds "  << tdcOnsetLeakage << " fC @ " << it+busyBxs << " bx (first free bx)" << std::endl;
	  newCharge[it+busyBxs] +=  tdcOnsetLeakage;
	}
    }
  
  //including the leakage from bunches in SARS ADC when not declared busy
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      if(busyFlags[it]) continue;
      float charge(chargeColl[it]);
      if(charge==0) continue;
      for(int ipulse=-2; ipulse<(int)(adcPulse_.size())-2; ipulse++)
	{
	  if(it+ipulse<0) continue;
	  if(it+ipulse>=(int)(newCharge.size())) continue;
	  if(busyFlags[it+ipulse]) continue;
	  newCharge[it+ipulse]+=charge*adcPulse_[(ipulse+2)];
	}
    }
  

  //set new ADCs and ToA
  for(int it=0; it<(int)(newCharge.size()); it++)
    {
      ////  if(debug) std::cout << chargeColl[it] << " -> " << newCharge[it] << " ";

      HGCSample newSample;
      if(busyFlags[it])
	{
	  if(newCharge[it]==0) newSample.set(false,true,0,0);
	  else                 newSample.set(true,true,toaFromToT[it],newCharge[it]/tdcLSB_fC_);
	}
      else
	{
	  newSample.set(newCharge[it]>adcThreshold_fC_,false,0,newCharge[it]/adcLSB_fC_);
	}
      dataFrame.setSample(it,newSample);
    }

  ////  if(debug) std::cout << std::endl;

}

