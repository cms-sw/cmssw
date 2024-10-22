
//HcalFeatureHFEMBit
//version 2.0

#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureHFEMBit.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

HcalFeatureHFEMBit::HcalFeatureHFEMBit(double ShortMinE,
                                       double LongMinE,
                                       double ShortLongCutSlope,
                                       double ShortLongCutOffset,
                                       const HcalDbService& conditions)
    : conditions_(conditions) {
  ShortMinE_ = ShortMinE;  //minimum energy deposited
  LongMinE_ = LongMinE;
  ShortLongCutSlope_ =
      ShortLongCutSlope;  // this is a the slope of the cut line related to energy deposited in short fibers vrs long fibers
  ShortLongCutOffset_ = ShortLongCutOffset;  // this is the offset of said line.
}

HcalFeatureHFEMBit::~HcalFeatureHFEMBit() {}

bool HcalFeatureHFEMBit::fineGrainbit(const HFDataFrame& shortDigi, const HFDataFrame& longDigi, int idx) const {
  float shortE = getE(shortDigi, idx);
  float longE = getE(longDigi, idx);

  if (shortE < ShortMinE_)
    return false;
  if (longE < LongMinE_)
    return false;

  return (shortE < (longE - ShortLongCutOffset_) * ShortLongCutSlope_);
}

bool HcalFeatureHFEMBit::fineGrainbit(const QIE10DataFrame& short1,
                                      const QIE10DataFrame& short2,
                                      const QIE10DataFrame& long1,
                                      const QIE10DataFrame& long2,
                                      bool validShort1,
                                      bool validShort2,
                                      bool validLong1,
                                      bool validLong2,
                                      int idx) const {
  float shortE = 0;
  if (validShort1)
    shortE += getE(short1, idx);
  if (validShort2)
    shortE += getE(short2, idx);
  if (validShort1 and validShort2)
    shortE *= .5;

  float longE = 0;
  if (validLong1)
    longE += getE(long1, idx);
  if (validLong2)
    longE += getE(long2, idx);
  if (validLong1 and validLong2)
    longE *= .5;

  if (shortE < ShortMinE_)
    return false;
  if (longE < LongMinE_)
    return false;

  return (shortE < (longE - ShortLongCutOffset_) * ShortLongCutSlope_);
}
