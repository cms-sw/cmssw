#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"



using namespace std;

//
template<class D>
HGCFEElectronics<D>::HGCFEElectronics(const edm::ParameterSet &ps) :
  toaMode_(WEIGHTEDBYE)
{
  tdcResolutionInNs_ = 1e-9; // set time resolution very small by default

  fwVersion_                      = ps.getParameter< uint32_t >("fwVersion");
  std::cout << "[HGCFEElectronics] running with version " << fwVersion_ << std::endl;
  if( ps.exists("adcPulse") )                       
    {
      adcPulse_  = ps.getParameter< std::vector<double> >("adcPulse");
      pulseAvgT_ = ps.getParameter< std::vector<double> >("pulseAvgT");
    }
  adcSaturation_fC_=-1.0;
  if( ps.exists("adcNbits") )
    {
      uint32_t adcNbits = ps.getParameter<uint32_t>("adcNbits");
      adcSaturation_fC_ = ps.getParameter<double>("adcSaturation_fC");
      adcLSB_fC_=adcSaturation_fC_/pow(2.,adcNbits);
      cout << "[HGCFEElectronics] " << adcNbits << " bit ADC defined"
	   << " with LSB=" << adcLSB_fC_ 
	   << " saturation to occur @ " << adcSaturation_fC_ << endl;
    }

  tdcSaturation_fC_=-1.0;
  if( ps.exists("tdcNbits") )
    {
      uint32_t tdcNbits = ps.getParameter<uint32_t>("tdcNbits");
      tdcSaturation_fC_ = ps.getParameter<double>("tdcSaturation_fC");
      tdcLSB_fC_=tdcSaturation_fC_/pow(2.,tdcNbits);
      cout << "[HGCFEElectronics] " << tdcNbits << " bit TDC defined with LSB=" << tdcLSB_fC_ << " saturation to occur @ " << tdcSaturation_fC_ << endl;
    }
  if( ps.exists("adcThreshold_fC") )                adcThreshold_fC_                = ps.getParameter<double>("adcThreshold_fC");
  if( ps.exists("tdcOnset_fC") )                    tdcOnset_fC_                    = ps.getParameter<double>("tdcOnset_fC");
  if( ps.exists("toaLSB_ns") )                      toaLSB_ns_                      = ps.getParameter<double>("toaLSB_ns");
  if( ps.exists("tdcChargeDrainParameterisation") ) tdcChargeDrainParameterisation_ = ps.getParameter< std::vector<double> >("tdcChargeDrainParameterisation");
  if( ps.exists("tdcResolutionInPs") )              tdcResolutionInNs_              = ps.getParameter<double>("tdcResolutionInPs")*1e-3; // convert to ns
  if( ps.exists("toaMode") )                        toaMode_                        = ps.getParameter<uint32_t>("toaMode");
}


//
template<class D>
void HGCFEElectronics<D>::runTrivialShaper(D &dataFrame,std::vector<float> &chargeColl)
{
  bool debug(false);
  
  //to enable debug uncomment me
  //for(int it=0; it<(int)(chargeColl.size()); it++) debug |= (chargeColl[it]>adcThreshold_fC_);
    
  if(debug) cout << "[runTrivialShaper]" << endl;
  
  //set new ADCs
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      //brute force saturation, maybe could to better with an exponential like saturation
      HGCSample newSample;
      uint32_t adc=floor( min(chargeColl[it],adcSaturation_fC_) / adcLSB_fC_ );
      newSample.set(chargeColl[it]>adcThreshold_fC_,false,0,adc);
      dataFrame.setSample(it,newSample);

      if(debug) cout << adc << " (" << chargeColl[it] << "/" << adcLSB_fC_ << ") ";
    }

  if(debug) { cout << endl; } // dataFrame.print(std::cout); }
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
      //debug|=(charge>adcThreshold_fC_);

      if(debug) std::cout << "\t Redistributing SARS ADC" << charge << " @ " << it;
      
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
      HGCSample newSample;

      //brute force saturation, maybe could to better with an exponential like saturation
      float saturatedCharge(min(newCharge[it],adcSaturation_fC_));
      newSample.set(newCharge[it]>adcThreshold_fC_,false,0,floor(saturatedCharge/adcLSB_fC_));
      dataFrame.setSample(it,newSample);      

      if(debug) std::cout << floor(saturatedCharge/adcLSB_fC_) << " (" << saturatedCharge << "/" << adcLSB_fC_ <<" ) " ;
    }
  
  if(debug) { std::cout << std::endl; } // dataFrame.print(std::cout); }
}

//
template<class D>
void HGCFEElectronics<D>::runShaperWithToT(D &dataFrame,std::vector<float> &chargeColl,std::vector<float> &toaColl, CLHEP::RandGauss* tdcReso)
{
  std::vector<bool>  busyFlags(chargeColl.size(),false),totFlags(chargeColl.size(),false);
  std::vector<float> newCharge(chargeColl.size(),0);
  std::vector<float> toaFromToT(chargeColl.size(),0);

  //make me true to debug
  bool debug(false);

  //first identify bunches which will trigger ToT
  if(debug) std::cout << "[runShaperWithToT]" << endl;  
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      //if already flagged as busy it can't be re-used to trigger the ToT
      if(busyFlags[it]) continue;

      //if below TDC onset will be handled by SARS ADC later
      float charge = chargeColl[it];
      if(charge < tdcOnset_fC_)  continue;

      //raise TDC mode
      float toa    = toaColl[it];
      totFlags[it]=true;

      if(debug) std::cout << "\t q=" << charge << " fC with <toa>=" << toa << " ns, triggers ToT @ " << it << std::endl;

      //compute total charge to be integrated and integration time 
      //needs a loop as ToT will last as long as there is charge to dissipate
      int busyBxs(0);
      float totalCharge(charge), finalToA(toa), integTime(0);
      while(true)
	{
	  //compute integration time in ns and # bunches
	  float newIntegTime(0);
	  float charge_kfC(totalCharge*1e-3);
	  if(charge_kfC<tdcChargeDrainParameterisation_[3]) 
	    newIntegTime=tdcChargeDrainParameterisation_[0]*pow(charge_kfC,2)+tdcChargeDrainParameterisation_[1]*charge_kfC+tdcChargeDrainParameterisation_[2];
	  else if(charge_kfC<tdcChargeDrainParameterisation_[7])
	    newIntegTime=tdcChargeDrainParameterisation_[4]*pow(charge_kfC-tdcChargeDrainParameterisation_[3],2)+tdcChargeDrainParameterisation_[5]*(charge_kfC-tdcChargeDrainParameterisation_[3])+tdcChargeDrainParameterisation_[6];
	  else
	    newIntegTime=tdcChargeDrainParameterisation_[8]*pow(charge_kfC-tdcChargeDrainParameterisation_[7],2)+tdcChargeDrainParameterisation_[9]*(charge_kfC-tdcChargeDrainParameterisation_[7])+tdcChargeDrainParameterisation_[10];

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
	  if(toaMode_==WEIGHTEDBYE) finalToA=toa*charge;

	  //add leakage from previous bunches in SARS ADC mode
	  for(int jt=0; jt<it; jt++)
	    {
	      if(totFlags[jt] || busyFlags[jt]) continue;

	      float chargeDep_jt(chargeColl[jt]);
	      if(chargeDep_jt==0) continue;

	      int deltaT=(it-jt);
	      if(deltaT+2>(int)(adcPulse_.size())) continue;

	      float leakCharge( adcPulse_[deltaT+2]*chargeDep_jt );
	      if(debug) std::cout << "\t\t leaking " << chargeDep_jt << " fC @ deltaT=-" << deltaT << " -> +" << leakCharge << " with avgT=" << pulseAvgT_[deltaT+2] << std::endl;

	      totalCharge  += leakCharge;
	      if(toaMode_==WEIGHTEDBYE) finalToA     += leakCharge*pulseAvgT_[deltaT+2];
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
	      if(toaMode_==WEIGHTEDBYE) finalToA    += extraCharge*toaColl[jt];
	    }
	  
	  //finalize ToA contamination
	  if(toaMode_==WEIGHTEDBYE) finalToA /= totalCharge;
	}

      toaFromToT[it] = tdcReso->fire(finalToA,tdcResolutionInNs_);
      newCharge[it]  = (totalCharge-tdcOnset_fC_);      

      if(debug) std::cout << "\t Final busy estimate="<< integTime << " ns = " << busyBxs << " bxs" << std::endl
			  << "\t Total integrated=" << totalCharge << " fC <toa>=" << toaFromToT[it] << " (raw=" << finalToA << ") ns " << std::endl; 
      
      //last fC (tdcOnset) are dissipated trough pulse
      if(it+busyBxs<(int)(newCharge.size())) 
	{
	  float deltaT2nextBx((busyBxs*25-integTime));
	  float tdcOnsetLeakage(tdcOnset_fC_*exp(-deltaT2nextBx/tdcChargeDrainParameterisation_[11]));
	  if(debug) std::cout << "\t Leaking remainder of TDC onset " << tdcOnset_fC_ 
			      << " fC, to be dissipated in " << deltaT2nextBx 
			      << " DeltaT/tau=" << deltaT2nextBx << " / " << tdcChargeDrainParameterisation_[11] 
			      << " ns, adds "  << tdcOnsetLeakage << " fC @ " << it+busyBxs << " bx (first free bx)" << std::endl;
	  newCharge[it+busyBxs] +=  tdcOnsetLeakage;
	}
    }
  
  //including the leakage from bunches in SARS ADC when not declared busy or in ToT
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      //if busy, charge has been already integrated
      if(totFlags[it] || busyFlags[it]) continue;
      float charge(chargeColl[it]);
      if(charge==0) continue;

      if(debug) std::cout << "\t SARS ADC pulse activated @ " << it << " : ";
      for(int ipulse=-2; ipulse<(int)(adcPulse_.size())-2; ipulse++)
	{
	  if(it+ipulse<0) continue;
	  if(it+ipulse>=(int)(newCharge.size())) continue;

	  //notice that if the channel is already busy,
	  //it has already been affected by the leakage of the SARS ADC
	  if(totFlags[it] || busyFlags[it+ipulse]) continue;
	  float chargeLeak=charge*adcPulse_[(ipulse+2)];
	  if(debug) std::cout << " | " << it+ipulse << " " << chargeLeak << "( " << charge << "->";
	  newCharge[it+ipulse]+=chargeLeak;
	  if(debug) std::cout << newCharge[it+ipulse] << ") ";
	}
      
      if(debug) std::cout << std::endl;
    }

  //set new ADCs and ToA
  if(debug) std::cout << "\t final result : ";
  for(int it=0; it<(int)(newCharge.size()); it++)
    {
      if(debug) std::cout << chargeColl[it] << " -> " << newCharge[it] << " ";

      HGCSample newSample;
      if(totFlags[it] || busyFlags[it])
	{
	  if(totFlags[it]) 
	    {
	      float finalToA(toaFromToT[it]);
	      while(finalToA<0)  finalToA+=25.;
	      while(finalToA>25) finalToA-=25;

	      //brute force saturation, maybe could to better with an exponential like saturation
	      float saturatedCharge(min(newCharge[it],tdcSaturation_fC_));	      
	      newSample.set(true,true,finalToA/toaLSB_ns_,floor(saturatedCharge/tdcLSB_fC_));
	    }
	  else
	    {
	      newSample.set(false,true,0,0);
	    }
	}
      else
	{
	   //brute force saturation, maybe could to better with an exponential like saturation
	      float saturatedCharge(min(newCharge[it],adcSaturation_fC_));
	  newSample.set(newCharge[it]>adcThreshold_fC_,false,0,saturatedCharge/adcLSB_fC_);
	}
      dataFrame.setSample(it,newSample);
    }

  if(debug) { std::cout << std::endl;} // dataFrame.print(std::cout); }
}

