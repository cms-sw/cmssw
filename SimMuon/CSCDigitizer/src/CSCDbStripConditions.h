#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"

class CSCDbStripConditions : public CSCStripConditions
{
public:
  CSCDbStripConditions();
  virtual ~CSCDbStripConditions();

  /// fetch the maps from the database
  virtual void initializeEvent(const edm::EventSetup & es);

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const;
  /// total calibration precision
  virtual float gainVariance(const CSCDetId & detId, int channel) const {return 0.005;}

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const;
  virtual float pedestalVariance(const CSCDetId & detId, int channel) const;

  virtual void crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight,
                 float & capacitive, float & resistive) const;

  void print() const;

private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);  
  // might change the channel # for ME1A
  static int dbIndex(const CSCDetId & id, int & channel);

  const CSCNoiseMatrix * theNoiseMatrix;
  
  const CSCGains * theGains;

  const CSCPedestals * thePedestals;

  const CSCcrosstalk * theCrosstalk;

  // nominal constant to give 100% crosstalk
  float theCapacitiveCrosstalk;
  float theResistiveCrosstalk;
};

#endif


