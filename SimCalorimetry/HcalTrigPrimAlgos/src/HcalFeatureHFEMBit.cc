
//HcalFeatureHFEMBit
//version 2.0

#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureHFEMBit.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"


HcalFeatureHFEMBit::HcalFeatureHFEMBit(double ShortMinE, double LongMinE,
        double ShortLongCutSlope, double ShortLongCutOffset, const HcalDbService& conditions) : conditions_(conditions)
{
    ShortMinE_ = ShortMinE; //minimum energy deposited
    LongMinE_ = LongMinE;
    ShortLongCutSlope_ = ShortLongCutSlope; // this is a the slope of the cut line related to energy deposited in short fibers vrs long fibers
    ShortLongCutOffset_ = ShortLongCutOffset; // this is the offset of said line.



}

HcalFeatureHFEMBit::~HcalFeatureHFEMBit() { }

float
HcalFeatureHFEMBit::getE(const HFDataFrame& f, int idx) const
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

bool
HcalFeatureHFEMBit::fineGrainbit(const HFDataFrame& shortDigi, const HFDataFrame& longDigi, int idx) const
{
    float shortE = getE(shortDigi, idx);
    float longE = getE(longDigi, idx);

    if (shortE < ShortMinE_)
       return false;
    if (longE < LongMinE_)
       return false;

    return (shortE < (longE - ShortLongCutOffset_) * ShortLongCutSlope_);
}


