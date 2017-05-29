#ifndef SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOENERGY_H
#define SIMCALORIMETRY_HCALZEROSUPPRESSIONALGOS_HCALZSALGOENERGY_H 1

#include "HcalZeroSuppressionAlgo.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

/** \class HcalZSAlgoEnergy
  *  
  * Simple amplitude-based zero suppression algorithm.  For each digi, add
  * up (samples) samples beginning with (start) sample.  Subtract the average
  * pedestal, and compare with digital threshold (in ADC counts).  The algorithm
  * can keep both positive and negative side fluctuations if "two sided" is enabled.
  *
  * \author J. Mans - Minnesota
  */
class HcalZSAlgoEnergy : public HcalZeroSuppressionAlgo {
public:
  HcalZSAlgoEnergy(bool markAndPass, int level, int start, int samples, bool twosided);
  virtual ~HcalZSAlgoEnergy() = default;
  void prepare(const HcalDbService* db);
  void done();
protected:
  virtual bool shouldKeep(const HBHEDataFrame& digi) const;
  virtual bool shouldKeep(const HODataFrame& digi) const;
  virtual bool shouldKeep(const HFDataFrame& digi) const;
  virtual bool shouldKeep(const QIE10DataFrame& digi) const;
  virtual bool shouldKeep(const QIE11DataFrame& digi) const;
private:
  int threshold_, firstsample_, samplecount_;
  bool twosided_;
  const HcalDbService* db_;
};

#endif
