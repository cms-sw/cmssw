#ifndef SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOREALISTIC_H
#define SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOREALISTIC_H 1

#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZeroSuppressionAlgo.h"

/** \class HcalZSAlgoRealistic
  *  
  * Simple amplitude-based zero suppression algorithm.  For each digi, add
  * up consecutive 2 samples in a slice of 10 time samples, beginning with (start) 
  * sample. If any of the sums are greater then the threshold, keep the event.
  *
  * $Date: 2008/10/07 16:47:58 $
  * $Revision: 1.1 $
  * \author S. Sengupta - Minnesota
  */
class HcalZSAlgoRealistic : public HcalZeroSuppressionAlgo {
public:
  HcalZSAlgoRealistic(ZSMode mode, int levelHB, int levelHE, int levelHO, int levelHF);

protected:
  virtual bool shouldKeep(const HBHEDataFrame& digi) const;
  virtual bool shouldKeep(const HODataFrame& digi) const;
  virtual bool shouldKeep(const HFDataFrame& digi) const;
private:
  int thresholdHF_, thresholdHB_, thresholdHE_, thresholdHO_;

};

#endif
