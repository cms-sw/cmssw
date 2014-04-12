#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include <cstdio>
#include <stdint.h>

using namespace std;


const uint32_t HcalTBTriggerData::bit_BeamTrigger                   = 1;
const uint32_t HcalTBTriggerData::bit_InSpillPedestalTrigger        = 2;
const uint32_t HcalTBTriggerData::bit_OutSpillPedestalTrigger       = 3;
const uint32_t HcalTBTriggerData::bit_LaserTrigger                  = 4;
const uint32_t HcalTBTriggerData::bit_spillIgnorantPedestalTrigger  = 5;
const uint32_t HcalTBTriggerData::bit_LEDTrigger                    = 6;

const uint32_t HcalTBTriggerData::bit_FakeTrigger                   = 15;

const uint32_t HcalTBTriggerData::bit_InSpill = 0x10;

  HcalTBTriggerData::HcalTBTriggerData() :
    runNumberSequenceId_(""),
    runNumber_(0),
    triggerWord_(0),
    triggerTimeUsec_(0),
    triggerTimeBase_(0),
    spillNumber_(0),
    orbitNumber_(0),
    bunchNumber_(0),
    eventNumber_(0),
    flagsDaqTtype_(0),
    algoBits3_(0),
    algoBits2_(0),
    algoBits1_(0),
    algoBits0_(0),
    techBits_(0),
    gps1234_(0),
    gps5678_(0) {
  }

  
  void HcalTBTriggerData::setStandardData(  uint32_t orbitNumber,
					    uint32_t eventNumber,
					    uint16_t bunchNumber,
					    uint32_t flags_daq_ttype,
					    uint32_t algo_bits_3,
					    uint32_t algo_bits_2,
					    uint32_t algo_bits_1,
					    uint32_t algo_bits_0,
					    uint32_t tech_bits,
					    uint32_t gps_1234,
					    uint32_t gps_5678) {

    orbitNumber_   = orbitNumber;
    eventNumber_   = eventNumber & 0x00ffffff; // only lower 24 bits active
    bunchNumber_   = bunchNumber;
    flagsDaqTtype_ = flags_daq_ttype;
    algoBits3_     = algo_bits_3;
    algoBits2_     = algo_bits_2;
    algoBits1_     = algo_bits_1;
    algoBits0_     = algo_bits_0;
    techBits_      = tech_bits;
    gps1234_       = gps_1234;
    gps5678_       = gps_5678;
  }

  void HcalTBTriggerData::setExtendedData(  uint32_t triggerWord,
					    uint32_t triggerTime_usec,
					    uint32_t triggerTime_base,
					    uint32_t spillNumber,
					    uint32_t runNumber,
					    const char *runNumberSequenceId ) {
    triggerWord_         = triggerWord;
    triggerTimeUsec_     = triggerTime_usec;
    triggerTimeBase_     = triggerTime_base;
    spillNumber_         = spillNumber;
    runNumber_           = runNumber;
    runNumberSequenceId_ = runNumberSequenceId;
  }

  ostream& operator<<(ostream& s, const HcalTBTriggerData& htbtd) {
    char str[50];
    s << htbtd.runNumberSequenceId() << ":";
    s << htbtd.runNumber() << ":";
    s << htbtd.eventNumber() << endl;

    s << "  Spill# =" << htbtd.spillNumber() << endl;
    s << "  Bunch# =" << htbtd.bunchNumber() << endl;
    s << "  Orbit# =" << htbtd.orbitNumber() << endl;

    sprintf(str, "  Trigger word = %08x\n", htbtd.triggerWord());
    s << str;

    int32_t  trigtimebase = (int32_t)htbtd.triggerTimeBase();
    uint32_t trigtimeusec = htbtd.triggerTimeUsec();

    // trim seconds off of usec and add to base
    trigtimebase += trigtimeusec/1000000;
    trigtimeusec %= 1000000;
    
    sprintf(str, "  Trigger time: %s", ctime((time_t *)&trigtimebase));
    s << str;
    sprintf(str, "                %d us\n", trigtimeusec);
    s << str;

    return s;
  }
