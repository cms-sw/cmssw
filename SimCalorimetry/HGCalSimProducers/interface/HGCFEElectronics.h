#ifndef _hgcfeelectronics_h_
#define _hgcfeelectronics_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

/**
   @class HGCFEElectronics
   @short models the behavior of the front-end electronics
 */

namespace hgc = hgc_digi;

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
  inline void runShaper(DFr &dataFrame, hgc::HGCSimHitData& chargeColl, 
                        hgc::HGCSimHitData& toa, int thickness, CLHEP::HepRandomEngine* engine)
  {    
    switch(fwVersion_)
      {
      case SIMPLE :  { runSimpleShaper(dataFrame,chargeColl, thickness);      break; }
      case WITHTOT : { runShaperWithToT(dataFrame,chargeColl,toa, thickness, engine); break; }
      default :      { runTrivialShaper(dataFrame,chargeColl, thickness);     break; }
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
  void runTrivialShaper(DFr &dataFrame, hgc::HGCSimHitData& chargeColl, int thickness);

  /**
     @short applies a shape to each time sample and propagates the tails to the subsequent time samples
   */
  void runSimpleShaper(DFr &dataFrame, hgc::HGCSimHitData& chargeColl, int thickness);

  /**
     @short implements pulse shape and switch to time over threshold including deadtime
   */
  void runShaperWithToT(DFr &dataFrame, hgc::HGCSimHitData& chargeColl, 
                        hgc::HGCSimHitData& toa, int thickness, CLHEP::HepRandomEngine* engine);

  /**
     @short returns how ToT will be computed
   */
  uint32_t toaMode() const { return toaMode_; }
  
  /**
     @short DTOR
   */
  ~HGCFEElectronics() {}

 private:
  
  //private members
  uint32_t fwVersion_;
  std::array<float,6> adcPulse_, pulseAvgT_;
  std::vector<float> tdcChargeDrainParameterisation_;
  float adcSaturation_fC_, adcLSB_fC_, tdcLSB_fC_, tdcSaturation_fC_,
    adcThreshold_fC_, tdcOnset_fC_, toaLSB_ns_, tdcResolutionInNs_; 
  uint32_t toaMode_;
  //caches
  std::array<bool,hgc::nSamples>  busyFlags, totFlags;
  hgc::HGCSimHitData newCharge, toaFromToT;
};

#endif
