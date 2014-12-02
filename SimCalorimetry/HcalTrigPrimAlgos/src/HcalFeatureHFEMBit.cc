
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

bool HcalFeatureHFEMBit::fineGrainbit(int ADCShort, HcalDetId Sid, int CapIdS, int ADCLong, HcalDetId Lid, int CapIdL) const//pass det id
{

    float ShortE = 0; //holds deposited energy
    float LongE = 0;


    HcalQIESample sQIESample(ADCShort, CapIdS, 1, 1);
    //makes a QIE sample for the short fiber.
    HFDataFrame shortf(Sid);
    shortf.setSize(1); //not planning on there being anything else here at this point in time so setting the size to 1 shouldn't matter
    shortf.setSample(0, sQIESample); //inputs data into digi.
    const HcalCalibrations& calibrations = conditions_.getHcalCalibrations(Sid);
    const HcalQIECoder* channelCoderS = conditions_.getHcalCoder(Sid);
    const HcalQIEShape* shapeS = conditions_.getHcalShape(channelCoderS);

    HcalCoderDb coders(*channelCoderS, *shapeS);

    CaloSamples tools;
    coders.adc2fC(shortf, tools);
    ShortE = (tools[0] - calibrations.pedestal(CapIdS)) * calibrations.respcorrgain(CapIdS);

    HcalQIESample lQIESample(ADCLong, CapIdL, 1, 1);
    HFDataFrame longf(Lid);
    longf.setSize(1);
    longf.setSample(0, lQIESample);
    const HcalCalibrations& calibrationL = conditions_.getHcalCalibrations(Lid);

    CaloSamples tool_l;

    const HcalQIECoder* channelCoderL = conditions_.getHcalCoder(Lid);
    const HcalQIEShape* shapeL = conditions_.getHcalShape(channelCoderL);

    HcalCoderDb coderL(*channelCoderL, *shapeL);

    coderL.adc2fC(longf, tool_l); // this fills tool_l[0] with linearized adc
    LongE = (tool_l[0] - calibrationL.pedestal(CapIdL)) * calibrationL.respcorrgain(CapIdL);

    
    // this actually does the cut
    if((ShortE < ((LongE)-(ShortLongCutOffset_)) * ShortLongCutSlope_) && LongE > LongMinE_ && ShortE > ShortMinE_) return true;
    else return false;
}


