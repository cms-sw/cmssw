#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"

class CSCDbStripConditions : public CSCStripConditions {
public:
  explicit CSCDbStripConditions(const edm::ParameterSet &pset);
  ~CSCDbStripConditions() override;

  /// fetch the maps from the database
  void initializeEvent(const edm::EventSetup &es) override;

  /// channels count from 1
  float gain(const CSCDetId &detId, int channel) const override;
  /// total calibration precision
  float gainSigma(const CSCDetId &detId, int channel) const override { return 0.005; }

  /// in ADC counts
  float pedestal(const CSCDetId &detId, int channel) const override;
  float pedestalSigma(const CSCDetId &detId, int channel) const override;

  void crosstalk(const CSCDetId &detId,
                 int channel,
                 double stripLength,
                 bool leftRight,
                 float &capacitive,
                 float &resistive) const override;

  /// check list of bad chambers from db
  bool isInBadChamber(const CSCDetId &id) const override;

private:
  void fetchNoisifier(const CSCDetId &detId, int istrip) override;

  CSCConditions theConditions;

  // nominal constant to give 100% crosstalk
  float theCapacitiveCrosstalk;
  // constant for resistive crosstalk scaling.
  //  Not really sure why it shouldn't be one.
  float theResistiveCrosstalkScaling;
  // converts DB gains to the gain we expect, 0.5 fC/ADC
  float theGainsConstant;
  bool doCorrelatedNoise_;
};

#endif
