#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/transform.h"

#include "vdt/vdtMath.h"

using namespace hgc_digi;

//
template<class DFr>
HGCFEElectronics<DFr>::HGCFEElectronics(const edm::ParameterSet &ps) :
  toaMode_(WEIGHTEDBYE)
{
  tdcResolutionInNs_ = 1e-9; // set time resolution very small by default

  fwVersion_                      = ps.getParameter< uint32_t >("fwVersion");
  edm::LogVerbatim("HGCFE") << "[HGCFEElectronics] running with version " << fwVersion_ << std::endl;
  if( ps.exists("adcPulse") )                       
    {
      auto temp = ps.getParameter< std::vector<double> >("adcPulse");
      for( unsigned i = 0; i < temp.size(); ++i ) {
        adcPulse_[i] = (float)temp[i];
      }
      // normalize adc pulse
      for( unsigned i = 0; i < adcPulse_.size(); ++i ) {
        adcPulse_[i] = adcPulse_[i]/adcPulse_[2];
      }
      temp = ps.getParameter< std::vector<double> >("pulseAvgT");
      for( unsigned i = 0; i < temp.size(); ++i ) {
        pulseAvgT_[i] = (float)temp[i];
      }
    }
  adcSaturation_fC_=-1.0;
  if( ps.exists("adcNbits") )
    {
      uint32_t adcNbits = ps.getParameter<uint32_t>("adcNbits");
      adcSaturation_fC_ = ps.getParameter<double>("adcSaturation_fC");
      adcLSB_fC_=adcSaturation_fC_/pow(2.,adcNbits);
      edm::LogVerbatim("HGCFE") 
         << "[HGCFEElectronics] " << adcNbits << " bit ADC defined"
	 << " with LSB=" << adcLSB_fC_ 
	 << " saturation to occur @ " << adcSaturation_fC_ << std::endl;
    }

  tdcSaturation_fC_=-1.0;
  if( ps.exists("tdcNbits") )
    {
      uint32_t tdcNbits = ps.getParameter<uint32_t>("tdcNbits");
      tdcSaturation_fC_ = ps.getParameter<double>("tdcSaturation_fC");
      tdcLSB_fC_=tdcSaturation_fC_/pow(2.,tdcNbits);
      edm::LogVerbatim("HGCFE") 
         << "[HGCFEElectronics] " << tdcNbits << " bit TDC defined with LSB=" 
         << tdcLSB_fC_ << " saturation to occur @ " << tdcSaturation_fC_ << std::endl;
    }
  if( ps.exists("adcThreshold_fC") )                adcThreshold_fC_                = ps.getParameter<double>("adcThreshold_fC");
  if( ps.exists("tdcOnset_fC") )                    tdcOnset_fC_                    = ps.getParameter<double>("tdcOnset_fC");
  if( ps.exists("toaLSB_ns") )                      toaLSB_ns_                      = ps.getParameter<double>("toaLSB_ns");
  if( ps.exists("tdcChargeDrainParameterisation") ) {
    for( auto val : ps.getParameter< std::vector<double> >("tdcChargeDrainParameterisation") ) {
      tdcChargeDrainParameterisation_.push_back((float)val);
    }
  }
  if( ps.exists("tdcResolutionInPs") )              tdcResolutionInNs_              = ps.getParameter<double>("tdcResolutionInPs")*1e-3; // convert to ns
  if( ps.exists("toaMode") )                        toaMode_                        = ps.getParameter<uint32_t>("toaMode");
}


//
template<class DFr>
void HGCFEElectronics<DFr>::runTrivialShaper(DFr &dataFrame, HGCSimHitData& chargeColl, int thickness)
{
  bool debug(false);
  
#ifdef EDM_ML_DEBUG  
  for(int it=0; it<(int)(chargeColl.size()); it++) debug |= (chargeColl[it]>adcThreshold_fC_);
#endif
    
  if(debug) edm::LogVerbatim("HGCFE") << "[runTrivialShaper]" << std::endl;
  
  //set new ADCs
 
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {      
      //brute force saturation, maybe could to better with an exponential like saturation      
      const uint32_t adc=std::floor( std::min(chargeColl[it],adcSaturation_fC_) / adcLSB_fC_ );
      HGCSample newSample;
      newSample.set(chargeColl[it]>thickness*adcThreshold_fC_,false,0,adc);
      dataFrame.setSample(it,newSample);

      if(debug) edm::LogVerbatim("HGCFE") << adc << " (" << chargeColl[it] << "/" << adcLSB_fC_ << ") ";
    }

  if(debug) { 
    std::ostringstream msg;
    dataFrame.print(msg);
    edm::LogVerbatim("HGCFE") << msg.str() << std::endl;
  } 
}

//
template<class DFr>
void HGCFEElectronics<DFr>::runSimpleShaper(DFr &dataFrame, HGCSimHitData& chargeColl, int thickness)
{
  //convolute with pulse shape to compute new ADCs
  newCharge.fill(0.f);
  bool debug(false);
  for(int it=0; it<(int)(chargeColl.size()); it++)
    {
      const float charge(chargeColl[it]);
      if(charge==0.f) continue;
            
#ifdef EDM_ML_DEBUG
      debug|=(charge>adcThreshold_fC_);
#endif

      if(debug) edm::LogVerbatim("HGCFE") << "\t Redistributing SARS ADC" << charge << " @ " << it;
      
      for(int ipulse=-2; ipulse<(int)(adcPulse_.size())-2; ipulse++)
	{
	  if(it+ipulse<0) continue;
	  if(it+ipulse>=(int)(dataFrame.size())) continue;
          const float chargeLeak=charge*adcPulse_[(ipulse+2)];
	  newCharge[it+ipulse]+= chargeLeak;
	  
	  if(debug) edm::LogVerbatim("HGCFE") << " | " << it+ipulse << " " << chargeLeak;
	}
      
      if(debug) edm::LogVerbatim("HGCFE") << std::endl;
    }

  //set new ADCs
    for(int it=0; it<(int)(newCharge.size()); it++)
    {
      //brute force saturation, maybe could to better with an exponential like saturation
      const float saturatedCharge(std::min(newCharge[it],adcSaturation_fC_));
      HGCSample newSample;
      newSample.set(newCharge[it]>thickness*adcThreshold_fC_,false,0,floor(saturatedCharge/adcLSB_fC_));
      dataFrame.setSample(it,newSample);      

      if(debug) edm::LogVerbatim("HGCFE") << std::floor(saturatedCharge/adcLSB_fC_) << " (" << saturatedCharge << "/" << adcLSB_fC_ <<" ) " ;
    }
  
  if(debug) { 
    std::ostringstream msg;
    dataFrame.print(msg);
    edm::LogVerbatim("HGCFE") << msg.str() << std::endl;
  }
}

//
template<class DFr>
void HGCFEElectronics<DFr>::runShaperWithToT(DFr &dataFrame, HGCSimHitData& chargeColl, HGCSimHitData& toaColl, int thickness, CLHEP::HepRandomEngine* engine)
{
  busyFlags.fill(false);
  totFlags.fill(false);
  newCharge.fill( 0.f );
  toaFromToT.fill( 0.f );

#ifdef EDM_ML_DEBUG
  constexpr bool debug_state(true);
#else
  constexpr bool debug_state(false);
#endif

  bool debug = debug_state;

  //first identify bunches which will trigger ToT
  //if(debug_state) edm::LogVerbatim("HGCFE") << "[runShaperWithToT]" << std::endl;  
  for(int it=0; it<(int)(chargeColl.size()); ++it)
    {
      debug = debug_state;
      //if already flagged as busy it can't be re-used to trigger the ToT
      if(busyFlags[it]) continue;

      //if below TDC onset will be handled by SARS ADC later
      float charge = chargeColl[it];
      if(charge < tdcOnset_fC_)  {
        debug = false;
        continue;
      }

      //raise TDC mode
      float toa    = toaColl[it];
      totFlags[it]=true;

      if(debug) edm::LogVerbatim("HGCFE") << "\t q=" << charge << " fC with <toa>=" << toa << " ns, triggers ToT @ " << it << std::endl;

      //compute total charge to be integrated and integration time 
      //needs a loop as ToT will last as long as there is charge to dissipate
      int busyBxs(0);
      float totalCharge(charge), finalToA(toa), integTime(0);
      while(true) {        
        //compute integration time in ns and # bunches
        //float newIntegTime(0);
        int poffset = 0;
        float charge_offset = 0.f;
        const float charge_kfC(totalCharge*1e-3);
        if(charge_kfC<tdcChargeDrainParameterisation_[3]) {
          //newIntegTime=tdcChargeDrainParameterisation_[0]*pow(charge_kfC,2)+tdcChargeDrainParameterisation_[1]*charge_kfC+tdcChargeDrainParameterisation_[2];
        } else if(charge_kfC<tdcChargeDrainParameterisation_[7]) {
          poffset = 4;
          charge_offset = tdcChargeDrainParameterisation_[3];          
          //newIntegTime=tdcChargeDrainParameterisation_[4]*pow(charge_kfC-tdcChargeDrainParameterisation_[3],2)+tdcChargeDrainParameterisation_[5]*(charge_kfC-tdcChargeDrainParameterisation_[3])+tdcChargeDrainParameterisation_[6];
        } else {
          poffset = 8;
          charge_offset = tdcChargeDrainParameterisation_[7];
          //newIntegTime=tdcChargeDrainParameterisation_[8]*pow(charge_kfC-tdcChargeDrainParameterisation_[7],2)+tdcChargeDrainParameterisation_[9]*(charge_kfC-tdcChargeDrainParameterisation_[7])+tdcChargeDrainParameterisation_[10];
        }
        const float charge_mod = charge_kfC - charge_offset;
        const float newIntegTime = ( ( tdcChargeDrainParameterisation_[poffset]*charge_mod + 
                                       tdcChargeDrainParameterisation_[poffset+1]            )*charge_mod +
                                       tdcChargeDrainParameterisation_[poffset+2] );
        
        const int newBusyBxs=std::floor(newIntegTime/25.f)+1;      
        
        //if no update is needed regarding the number of bunches,
        //then the ToT integration time has converged
        integTime=newIntegTime;
        if(newBusyBxs==busyBxs) break;
        
        //update charge integrated during ToT
        if(debug)
          {
            if(busyBxs==0) edm::LogVerbatim("HGCFE") << "\t Intial busy estimate="<< integTime << " ns = " << newBusyBxs << " bxs" << std::endl;
            else           edm::LogVerbatim("HGCFE") << "\t ...integrated charge overflows initial busy estimate, interating again" << std::endl;
          }
        
        //update number of busy bunches
        busyBxs=newBusyBxs;
        
        //reset charge to be integrated
        totalCharge=charge;
        if(toaMode_==WEIGHTEDBYE) finalToA=toa*charge;
        
        //add leakage from previous bunches in SARS ADC mode
        for(int jt=0; jt<it; ++jt)
          {
            const unsigned int deltaT=(it-jt);
            if((deltaT+2) >=  adcPulse_.size() || chargeColl[jt]==0.f || totFlags[jt] || busyFlags[jt]) continue;
            
            const float leakCharge = chargeColl[jt]*adcPulse_[deltaT+2];
            totalCharge += leakCharge;
            if(toaMode_==WEIGHTEDBYE) finalToA += leakCharge*pulseAvgT_[deltaT+2];
            
            if(debug) edm::LogVerbatim("HGCFE") << "\t\t leaking " << chargeColl[jt] << " fC @ deltaT=-" << deltaT << " -> +" << leakCharge << " with avgT=" << pulseAvgT_[deltaT+2] << std::endl;
          }
        
        //add contamination from posterior bunches
        for(int jt=it+1; jt<it+busyBxs && jt<dataFrame.size() ; ++jt) 
          { 
            //this charge will be integrated in TDC mode
            //disable for SARS ADC
            busyFlags[jt]=true; 
            
            const float extraCharge=chargeColl[jt];
            if(extraCharge==0.f) continue;
            if(debug) edm::LogVerbatim("HGCFE") << "\t\t adding " << extraCharge << " fC @ deltaT=+" << (jt-it) << std::endl; 
            
            totalCharge += extraCharge;
            if(toaMode_==WEIGHTEDBYE) finalToA    += extraCharge*toaColl[jt];
          }
        
        //finalize ToA contamination
        if(toaMode_==WEIGHTEDBYE) finalToA /= totalCharge;
      }

      toaFromToT[it] = CLHEP::RandGaussQ::shoot(engine,finalToA,tdcResolutionInNs_);
      newCharge[it]  = (totalCharge-tdcOnset_fC_);      
      
      if(debug) edm::LogVerbatim("HGCFE") << "\t Final busy estimate="<< integTime << " ns = " << busyBxs << " bxs" << std::endl
			                  << "\t Total integrated=" << totalCharge << " fC <toa>=" << toaFromToT[it] << " (raw=" << finalToA << ") ns " << std::endl; 
      
      //last fC (tdcOnset) are dissipated trough pulse
      if(it+busyBxs<(int)(newCharge.size())) 
	{
	  const float deltaT2nextBx((busyBxs*25-integTime));
	  const float tdcOnsetLeakage(tdcOnset_fC_*vdt::fast_expf(-deltaT2nextBx/tdcChargeDrainParameterisation_[11]));
	  if(debug) edm::LogVerbatim("HGCFE") << "\t Leaking remainder of TDC onset " << tdcOnset_fC_ 
			                      << " fC, to be dissipated in " << deltaT2nextBx 
			                      << " DeltaT/tau=" << deltaT2nextBx << " / " << tdcChargeDrainParameterisation_[11] 
			                      << " ns, adds "  << tdcOnsetLeakage << " fC @ " << it+busyBxs << " bx (first free bx)" << std::endl;
	  newCharge[it+busyBxs] +=  tdcOnsetLeakage;
	}
    }

  //including the leakage from bunches in SARS ADC when not declared busy or in ToT
  auto runChargeSharing = [&]() {
    int ipulse = 0;
    for(int it=0; it<(int)(chargeColl.size()); ++it)
      {
        //if busy, charge has been already integrated
        //if(debug) edm::LogVerbatim("HGCFE") << "\t SARS ADC pulse activated @ " << it << " : ";
        if( !totFlags[it] & !busyFlags[it] ) {
          const int start = std::max(0,2-it);
          const int stop  = std::min((int)adcPulse_.size(),(int)newCharge.size()-it+2);          
          for(ipulse = start; ipulse < stop; ++ipulse) {
            const int itoffset = it + ipulse - 2; 
            //notice that if the channel is already busy,
            //it has already been affected by the leakage of the SARS ADC
            //if(totFlags[itoffset] || busyFlags[itoffset]) continue;
            if( !totFlags[itoffset] & !busyFlags[itoffset] ) {
              newCharge[itoffset] += chargeColl[it]*adcPulse_[ipulse];
            }
            //if(debug) edm::LogVerbatim("HGCFE") << " | " << itoffset << " " << chargeColl[it]*adcPulse_[ipulse] << "( " << chargeColl[it] << "->";
            //if(debug) edm::LogVerbatim("HGCFE") << newCharge[itoffset] << ") ";
          }
        }
        
        if(debug) edm::LogVerbatim("HGCFE") << std::endl;
      }
  };
  runChargeSharing();

  //set new ADCs and ToA
  if(debug) edm::LogVerbatim("HGCFE") << "\t final result : ";
  const float adj_thresh = thickness*adcThreshold_fC_;
  for(int it=0; it<(int)(newCharge.size()); it++)
    {
      if(debug) edm::LogVerbatim("HGCFE") << chargeColl[it] << " -> " << newCharge[it] << " ";

      HGCSample newSample;
      if(totFlags[it] || busyFlags[it])
	{
	  if(totFlags[it]) 
	    {
	      float finalToA(toaFromToT[it]);
	      while(finalToA < 0.f)  finalToA+=25.f;
	      while(finalToA > 25.f) finalToA-=25.f;

	      //brute force saturation, maybe could to better with an exponential like saturation
	      const float saturatedCharge(std::min(newCharge[it],tdcSaturation_fC_));	      
	      newSample.set(true,true,(uint16_t)(finalToA/toaLSB_ns_),(uint16_t)(std::floor(saturatedCharge/tdcLSB_fC_)));
	    }
	  else
	    {
	      newSample.set(false,true,0,0);
	    }
	}
      else
	{
	   //brute force saturation, maybe could to better with an exponential like saturation
          const float saturatedCharge(std::min(newCharge[it],adcSaturation_fC_));
	  newSample.set(newCharge[it]>adj_thresh,false,(uint16_t)0,(uint16_t)(std::floor(saturatedCharge/adcLSB_fC_)));
	}
      dataFrame.setSample(it,newSample);
    }

  if(debug) { 
    std::ostringstream msg;
    dataFrame.print(msg);
    edm::LogVerbatim("HGCFE") << msg.str() << std::endl;
  }  
}

// cause the compiler to generate the appropriate code
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
template class HGCFEElectronics<HGCEEDataFrame>;
template class HGCFEElectronics<HGCBHDataFrame>;

