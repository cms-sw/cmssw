#ifndef _hgcfeelectronics_h_
#define _hgcfeelectronics_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

/**
   @class HGCFEElectronics
   @short models the behavior of the front-end electronics
 */

namespace hgc = hgc_digi;

template <class DFr>
class HGCFEElectronics {
public:
  enum HGCFEElectronicsFirmwareVersion { TRIVIAL, SIMPLE, WITHTOT };
  enum HGCFEElectronicsTOTMode { WEIGHTEDBYE, SIMPLETHRESHOLD };

  /**
     @short CTOR
   */
  HGCFEElectronics(const edm::ParameterSet& ps);

  /**
     @short switches according to the firmware version
   */
  inline void runShaper(DFr& dataFrame,
                        hgc::HGCSimHitData& chargeColl,
                        hgc::HGCSimHitData& toa,
                        CLHEP::HepRandomEngine* engine,
                        uint32_t thrADC = 0,
                        float lsbADC = -1,
                        uint32_t gainIdx = 0,
                        float maxADC = -1,
                        int thickness = 1) {
    switch (fwVersion_) {
      case SIMPLE: {
        runSimpleShaper(dataFrame, chargeColl, thrADC, lsbADC, gainIdx, maxADC);
        break;
      }
      case WITHTOT: {
        runShaperWithToT(dataFrame, chargeColl, toa, engine, thrADC, lsbADC, gainIdx, maxADC, thickness);
        break;
      }
      default: {
        runTrivialShaper(dataFrame, chargeColl, thrADC, lsbADC, gainIdx, maxADC);
        break;
      }
    }
  }

  void SetNoiseValues(const std::vector<float>& noise_fC) {
    noise_fC_.insert(noise_fC_.end(), noise_fC.begin(), noise_fC.end());
  };

  float getTimeJitter(float totalCharge, int thickness) {
    float A2 = jitterNoise2_ns_.at(thickness - 1);
    float C2 = jitterConstant2_ns_.at(thickness - 1);
    float X2 = pow((totalCharge / noise_fC_.at(thickness - 1)), 2.);
    float jitter2 = A2 / X2 + C2;
    return sqrt(jitter2);
  };

  /**
     @short returns the LSB in MIP currently configured
  */
  float getADClsb() { return adcLSB_fC_; }
  float getTDClsb() { return tdcLSB_fC_; }
  int getTargetMipValue() { return targetMIPvalue_ADC_; }
  float getADCThreshold() { return adcThreshold_fC_; }
  float getTDCOnset() { return tdcOnset_fC_; }
  std::array<float, 3> getTDCForToAOnset() { return tdcForToAOnset_fC_; }
  void setADClsb(float newLSB) { adcLSB_fC_ = newLSB; }

  /**
     @short converts charge to digis without pulse shape
   */
  void runTrivialShaper(
      DFr& dataFrame, hgc::HGCSimHitData& chargeColl, uint32_t thrADC, float lsbADC, uint32_t gainIdx, float maxADC);

  /**
     @short applies a shape to each time sample and propagates the tails to the subsequent time samples
   */
  void runSimpleShaper(
      DFr& dataFrame, hgc::HGCSimHitData& chargeColl, uint32_t thrADC, float lsbADC, uint32_t gainIdx, float maxADC);

  /**
     @short implements pulse shape and switch to time over threshold including deadtime
   */
  void runShaperWithToT(DFr& dataFrame,
                        hgc::HGCSimHitData& chargeColl,
                        hgc::HGCSimHitData& toa,
                        CLHEP::HepRandomEngine* engine,
                        uint32_t thrADC,
                        float lsbADC,
                        uint32_t gainIdx,
                        float maxADC,
                        int thickness);

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
  std::array<float, 6> adcPulse_, pulseAvgT_;
  std::array<float, 3> tdcForToAOnset_fC_;
  std::vector<float> tdcChargeDrainParameterisation_;
  float adcSaturation_fC_, adcLSB_fC_, tdcLSB_fC_, tdcSaturation_fC_, adcThreshold_fC_, tdcOnset_fC_, toaLSB_ns_,
      tdcResolutionInNs_;
  uint32_t targetMIPvalue_ADC_;
  std::array<float, 3> jitterNoise2_ns_, jitterConstant2_ns_;
  std::vector<float> noise_fC_;
  uint32_t toaMode_;
  bool thresholdFollowsMIP_;
  //caches
  std::array<bool, hgc::nSamples> busyFlags, totFlags, toaFlags;
  hgc::HGCSimHitData newCharge, toaFromToT;
};

#endif
