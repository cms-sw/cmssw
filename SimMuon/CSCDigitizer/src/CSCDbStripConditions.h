#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCDbStripConditions : public CSCStripConditions
{
public:
  explicit CSCDbStripConditions(const edm::ParameterSet & pset);
  virtual ~CSCDbStripConditions();

  /// fetch the maps from the database
  virtual void initializeEvent(const edm::EventSetup & es);

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const;
  /// total calibration precision
  virtual float gainSigma(const CSCDetId & detId, int channel) const {return 0.005;}

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const;
  virtual float pedestalSigma(const CSCDetId & detId, int channel) const;

  virtual void crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight,
                 float & capacitive, float & resistive) const;

  /// check list of bad chambers from db
  virtual bool isInBadChamber( const CSCDetId& id ) const;

private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);  

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


