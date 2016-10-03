#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included 1

#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureBit.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

class HcalFeatureHFEMBit : public HcalFeatureBit {
   public:
      HcalFeatureHFEMBit(double ShortMinE, double LongMinE, double ShortLongCutSlope, double ShortLongCutOffset, const HcalDbService& conditions);
      ~HcalFeatureHFEMBit();

      // Provides FG bit based on energy cuts in long & short fibers.
      virtual bool fineGrainbit(const HFDataFrame& shortDigi, const HFDataFrame& longDigi, int idx) const;
   private:
      float getE(const HFDataFrame& f, int idx) const;

      double ShortMinE_, LongMinE_, ShortLongCutSlope_, ShortLongCutOffset_;
      const HcalDbService& conditions_;
};
#endif
