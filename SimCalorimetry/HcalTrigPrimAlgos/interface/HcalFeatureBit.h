#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"

class HcalFeatureBit {
public:
  HcalFeatureBit() {}
  virtual ~HcalFeatureBit() {
  }  //the virutal function is responcible for applying a cut based on a linear relationship of the energy
  //deposited in the short vers long fibers.
  virtual bool fineGrainbit(const QIE10DataFrame& short1,
                            const QIE10DataFrame& short2,
                            const QIE10DataFrame& long1,
                            const QIE10DataFrame& long2,
                            bool validShort1,
                            bool validShort2,
                            bool validLong1,
                            bool validLong2,
                            int idx) const = 0;
  virtual bool fineGrainbit(const HFDataFrame& shortDigi, const HFDataFrame& longDigi, int idx) const = 0;
};
#endif
