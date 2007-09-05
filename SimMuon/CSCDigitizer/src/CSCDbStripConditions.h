#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"

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
  virtual float gainSigma(const CSCDetId & detId, int channel) const {return 0.005;}

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const;
  virtual float pedestalSigma(const CSCDetId & detId, int channel) const;

  virtual void crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight,
                 float & capacitive, float & resistive) const;

  void print() const;

private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);  

  CSCDBPedestals::Item pedestalObject(const CSCDetId & detId, int channel) const;

  // might change the channel # for ME1A
  static int dbIndex(const CSCDetId & id, int & channel);

  const CSCDBNoiseMatrix * theNoiseMatrix;
  
  const CSCDBGains * theGains;

  const CSCDBPedestals * thePedestals;

  const CSCDBCrosstalk * theCrosstalk;

  // nominal constant to give 100% crosstalk
  float theCapacitiveCrosstalk;
  // converts DB gains to the gain we expect, 0.5 fC/ADC
  float theGainsConstant;
};

#endif


