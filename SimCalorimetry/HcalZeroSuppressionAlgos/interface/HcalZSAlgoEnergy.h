#ifndef SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOENERGY_H
#define SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOENERGY_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

/** \class HcalZSAlgoEnergy
  *  
  * Simple amplitude-based zero suppression algorithm.  For each digi, add
  * up (samples) samples beginning with (start) sample.  Subtract the average
  * pedestal, and compare with digital threshold (in ADC counts).  The algorithm
  * can keep both positive and negative side fluctuations if "two sided" is enabled.
  *
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class HcalZSAlgoEnergy {
public:
  HcalZSAlgoEnergy(int level, int start, int samples, bool twosided);
  void suppress(const HcalDbService& db, const HBHEDigiCollection& inp, HBHEDigiCollection& outp) const;
  void suppress(const HcalDbService& db, const HODigiCollection& inp, HODigiCollection& outp) const;
  void suppress(const HcalDbService& db, const HFDigiCollection& inp, HFDigiCollection& outp) const;
private:
  int threshold_, firstsample_, samplecount_;
  bool twosided_;
};

#endif
