#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "FWCore/Framework/interface/EventSetup.h"

class CSCDbStripConditions : public CSCStripConditions
{
public:
  CSCDbStripConditions();

  /// fetch the maps from the database
  void read(const edm::EventSetup & es);

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const;
  virtual float gainVariance(const CSCDetId & detId, int channel) const {return 0.03;}

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const;


private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);  
  static int dbIndex(const CSCDetId & id);

  const CSCNoiseMatrix * theNoiseMatrix;
  
  const CSCGains * theGains;

  const CSCPedestals * thePedestals;
};

#endif


