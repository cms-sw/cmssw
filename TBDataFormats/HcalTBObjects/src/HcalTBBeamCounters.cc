#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"

using namespace std;

  HcalTBBeamCounters::HcalTBBeamCounters() :
    VMadc_(0),V3adc_(0),V6adc_(0),VH1adc_(0),VH2adc_(0),VH3adc_(0),VH4adc_(0),
    CK2adc_(0),CK3adc_(0),SciVLEadc_(0),Sci521adc_(0),Sci528adc_(0),
    S1_(0),S2_(0),S3_(0),S4_(0){}

  void HcalTBBeamCounters::setADCs  (uint16_t VMadc,uint16_t V3adc,uint16_t V6adc,
				     uint16_t VH1adc ,uint16_t VH2adc,uint16_t VH3adc,uint16_t VH4adc,
				     uint16_t CK2adc,uint16_t CK3adc,uint16_t SciVLEadc,
				     uint16_t Sci521adc,uint16_t Sci528adc,
                                     uint16_t S1,uint16_t S2,uint16_t S3,uint16_t S4)

  {
    VMadc_ = VMadc % 0x4000;
    V3adc_ = V3adc % 0x4000;
    V6adc_ = V6adc % 0x4000;
    VH1adc_ = VH1adc % 0x4000;
    VH2adc_ = VH2adc % 0x4000;
    VH3adc_ = VH3adc % 0x4000;
    VH4adc_ = VH4adc % 0x4000;
    CK2adc_ = CK2adc % 0x4000;
    CK3adc_ = CK3adc % 0x4000;
    SciVLEadc_ = SciVLEadc % 0x4000;
    Sci521adc_ = Sci521adc % 0x4000;
    Sci528adc_ = Sci528adc % 0x4000;
    S1_ = S1 % 0x4000;
    S2_ = S2 % 0x4000;
    S3_ = S3 % 0x4000;
    S4_ = S4 % 0x4000;
  }

  ostream& operator<<(ostream& s, const HcalTBBeamCounters& htbcnt) {

    s << "VM adc     = " << htbcnt.VMadc() << endl;

    return s;
  }
