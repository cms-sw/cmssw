#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included 1

#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureBit.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

class HcalFeatureHFEMBit : public HcalFeatureBit {
   public:
      HcalFeatureHFEMBit(double ShortMinE, double LongMinE, double ShortLongCutSlope, double ShortLongCutOffset, const HcalDbService& conditions);
      ~HcalFeatureHFEMBit();

      // Provides FG bit based on energy cuts in long & short fibers.
      virtual bool fineGrainbit(
            const QIE10DataFrame& short1,
            const QIE10DataFrame& short2,
            const QIE10DataFrame& long1,
            const QIE10DataFrame& long2,
            bool validShort1,
            bool validShort2,
            bool validLong1,
            bool validLong2,
            int idx) const override;
      virtual bool fineGrainbit(
            const HFDataFrame& shortDigi,
            const HFDataFrame& longDigi,
            int idx) const override;
   private:
      template<typename T>
      float getE(const T& f, int idx) const;

      double ShortMinE_, LongMinE_, ShortLongCutSlope_, ShortLongCutOffset_;
      const HcalDbService& conditions_;
};

template<typename T>
float
HcalFeatureHFEMBit::getE(const T& f, int idx) const
{
   const HcalDetId id(f.id());
   const HcalCalibrations& calibrations = conditions_.getHcalCalibrations(id);
   const auto* coder = conditions_.getHcalCoder(id);
   const auto* shape = conditions_.getHcalShape(coder);

   HcalCoderDb db(*coder, *shape);
   CaloSamples samples;
   db.adc2fC(f, samples);

   auto ped = calibrations.pedestal(f[idx].capid());
   auto corr = calibrations.respcorrgain(f[idx].capid());

   return (samples[idx] - ped) * corr;
}

#endif
