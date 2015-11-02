#ifndef _hgcfeelectronics_h_
#define _hgcfeelectronics_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandGaussQ.h"

/**
   @class HGCFEElectronics
   @short models the behavior of the front-end electronics
 */

template <class DFr>
class HGCFEElectronics
{
 public:

  enum HGCFEElectronicsFirmwareVersion { TRIVIAL, SIMPLE, WITHTOT };
  enum HGCFEElectronicsTOTMode { WEIGHTEDBYE, SIMPLETHRESHOLD };

  /**
     @short CTOR
   */
  HGCFEElectronics(const edm::ParameterSet &ps);

  /**
     @short switches according to the firmware version
   */
  inline void runShaper(DFr &dataFrame,std::vector<float> &chargeColl,std::vector<float> &toa, CLHEP::HepRandomEngine* engine)
  {    
    switch(fwVersion_)
      {
      case SIMPLE :  { runSimpleShaper(dataFrame,chargeColl);      break; }
      case WITHTOT : { runShaperWithToT(dataFrame,chargeColl,toa,engine); break; }
      default :      { runTrivialShaper(dataFrame,chargeColl);     break; }
      }
  }

  /**
     @short returns the LSB in MIP currently configured
  */
  float getADClsb()       { return adcLSB_fC_;       }
  float getTDClsb()       { return tdcLSB_fC_;       }  
  float getADCThreshold() { return adcThreshold_fC_; }
  float getTDCOnset()     { return tdcOnset_fC_;     }
  void setADClsb(float newLSB) { adcLSB_fC_=newLSB; }

  /**
     @short converts charge to digis without pulse shape
   */
  void runTrivialShaper(DFr &dataFrame,std::vector<float> &chargeColl);

  /**
     @short applies a shape to each time sample and propagates the tails to the subsequent time samples
   */
  void runSimpleShaper(DFr &dataFrame,std::vector<float> &chargeColl);

  /**
     @short implements pulse shape and switch to time over threshold including deadtime
   */
  void runShaperWithToT(DFr &dataFrame,std::vector<float> &chargeColl,std::vector<float> &toa, CLHEP::HepRandomEngine* engine);

  /**
     @short returns how ToT will be computed
   */
  uint32_t toaMode() const { return toaMode_; }
  
  void resetCaches();

  /**
     @short DTOR
   */
  ~HGCFEElectronics() {}

 private:
  
  //private members
  uint32_t fwVersion_;
  std::vector<double> adcPulse_,pulseAvgT_, tdcChargeDrainParameterisation_;
  float adcSaturation_fC_, adcLSB_fC_, tdcLSB_fC_, tdcSaturation_fC_,
    adcThreshold_fC_, tdcOnset_fC_, toaLSB_ns_, tdcResolutionInNs_; 
  uint32_t toaMode_;
  //caches
  std::vector<bool>  busyFlags, totFlags;
  std::vector<float> newCharge, toaFromToT;
};

#endif
