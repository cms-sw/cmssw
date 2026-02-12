#ifndef __SimFastTiming_FastTimingCommon_BTLElectronicsSim_h__
#define __SimFastTiming_FastTimingCommon_BTLElectronicsSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/MTDCommonData/interface/BTLElectronicsMapping.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLDigiSoA/interface/BTLDigiHostCollection.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

namespace mtd = mtd_digitizer;

namespace CLHEP {
  class HepRandomEngine;
}

class BTLElectronicsSim {
public:
  BTLElectronicsSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

  ~BTLElectronicsSim();

  void getEvent(const edm::Event& evt) {}

  void getEventSetup(const edm::EventSetup& evt) {}

  void run(const mtd::MTDSimHitDataAccumulator& input, BTLDigiCollection& output, CLHEP::HepRandomEngine* hre) const;

  void run(const mtd::MTDSimHitDataAccumulator& input,
           BTLDigiCollection& output,
           btldigi::BTLDigiHostCollection& outputSoA,
           CLHEP::HepRandomEngine* hre) const;

  void runTrivialShaper(BTLDataFrame& dataFrame,
                        const float (&charge)[2],
                        const float (&toa1)[2],
                        const float (&toa2)[2],
                        const uint8_t row,
                        const uint8_t col) const;

  bool checkValidHit(const BTLDataFrame& rawDataFrame) const;

  void updateOutput(BTLDigiCollection& coll, const BTLDataFrame& rawDataFrame) const;

  void updateOutputSoA(btldigi::BTLDigiHostCollection& coll,
                       int hitIndex,
                       uint32_t rawId,
                       const float (&charge_adc)[2],
                       const float (&toa1)[2],
                       const float (&toa2)[2],
                       const uint8_t col,
                       const uint16_t BC0count) const;

  static constexpr int dfSIZE = 2;

  BTLElectronicsMapping elMap_ = BTLElectronicsMapping(BTLDetId::CrysLayout::v4);

private:
  float rearming_time(const float& time, const float& npe) const;

  float pulse_tbranch_uA(const float& npe) const;

  float pulse_ebranch_uA(const float& npe) const;

  float time_at_Thr1Rise(const float& npe) const;

  float time_at_Thr2Rise(const float& npe) const;

  float time_over_Thr1(const float& npe) const;

  float sigma_stochastic(const float& npe) const;

  float sigma_DCR(const float& npe) const;

  float sigma_electronics(const float& npe) const;

  float pulse_q(const float& npe) const;

  float pulse_qRes(const float& npe) const;

  uint16_t timetoTcoarse(float time, const uint16_t mask) const;

  uint16_t timetoTfine(float time, const uint16_t tcoarse) const;

  uint16_t chargetoQfine(float charge, float toa1, uint16_t EOI) const;

  static constexpr float sqrt2_ = 1.41421356f;

  static constexpr float tofhirClock_ = 6.25f;  // [ns]
  static constexpr uint32_t adcBitSaturation_ = 1023;
  static constexpr uint32_t tdcBitSaturation_ = 1023;
  static constexpr float tdcLSB_ns_ = 0.020;  // [ns]

  static constexpr uint16_t T1coarseMask = 0x7FFF;  // 15 bits for T1 coarse time
  static constexpr uint16_t T2coarseMask = 0x2FF;   // 10 bits for T2 coarse time
  // static constexpr uint16_t TfineShift = 5;   // 10 bits for fine time

  // tdc calibration parameters
  // To be replaced by CondFormat when available
  static constexpr float a0_ = 57.244545;
  static constexpr float a1_ = 511.27832;
  static constexpr float a2_ = -7.8838577;
  static constexpr float t0_ = -0.048343264;

  // qdc calibration parameters
  // To be replaced by CondFormat when available
  static constexpr float p0_ = 49.542229;
  static constexpr float p1_ = -0.323424;
  static constexpr float p2_ = 0.062578;
  static constexpr float p3_ = -0.002484;
  static constexpr float p4_ = 0.0;
  static constexpr float p5_ = 0.0;
  static constexpr float p6_ = 0.0;
  static constexpr float p7_ = 0.0;
  static constexpr float p8_ = 0.0;
  static constexpr float p9_ = 0.0;

  static constexpr uint32_t numberOfRUs_ = 432;
  std::array<float, numberOfRUs_>* smearingClockRU_;

  const float bxTime_;
  const float lcepositionSlope_;
  const float sigmaLCEpositionSlope_;
  const float pulseT2Threshold_;
  const float pulseEThreshold_;
  const uint32_t channelRearmMode_;
  const float channelRearmNClocks_;
  const float t1Delay_;
  const float sipmGain_;
  const std::vector<double> paramPulseTbranchA_;
  const std::vector<double> paramPulseEbranchA_;
  const std::vector<double> paramThr1Rise_;
  const std::vector<double> paramThr2Rise_;
  const std::vector<double> paramTimeOverThr1_;
  const bool smearTimeForOOTtails_;
  const float scintillatorRiseTime_;
  const float scintillatorDecayTime_;
  const std::vector<double> stocasticParam_;
  const float darkCountRate_;
  const std::vector<double> paramDCR_;
  const float sigmaElectronicNoise_;
  const std::vector<double> paramSR_;
  const float sigmaTDC_;
  const float sigmaClockGlobal_;
  const float sigmaClockRU_;
  const std::vector<double> paramPulseQ_;
  const std::vector<double> paramPulseQRes_;
  const uint32_t integrationTimeFixed_;
  const float corrCoeff_;
  const float cosPhi_;
  const float sinPhi_;
  const float scintillatorDecayTimeInv_;
  const float sigmaConst2_;

  const bool debug_;
};

#endif
