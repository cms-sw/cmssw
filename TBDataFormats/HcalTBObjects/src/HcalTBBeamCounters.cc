#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"

using namespace std;

  HcalTBBeamCounters::HcalTBBeamCounters() :
    VMadc_(0),V3adc_(0),V6adc_(0),VH1adc_(0),VH2adc_(0),VH3adc_(0),VH4adc_(0),Ecal7x7_(0.),
    Sci521adc_(0),Sci528adc_(0),CK1adc_(0),CK2adc_(0),CK3adc_(0),SciVLEadc_(0),
    S1adc_(0),S2adc_(0),S3adc_(0),S4adc_(0),VMFadc_(0),VMBadc_(0),
    VM1adc_(0),VM2adc_(0),VM3adc_(0),VM4adc_(0),VM5adc_(0),VM6adc_(0),VM7adc_(0),VM8adc_(0),
    TOF1Sadc_(0),TOF1Jadc_(0),TOF2Sadc_(0),TOF2Jadc_(0),BH1adc_(0),BH2adc_(0),BH3adc_(0),BH4adc_(0){}

  void HcalTBBeamCounters::setADCs04  (double VMadc,double V3adc,double V6adc,
				     double VH1adc ,double VH2adc,double VH3adc,double VH4adc,
				     double CK2adc,double CK3adc,double SciVLEadc,
				     double Sci521adc,double Sci528adc,
                                     double S1adc,double S2adc,double S3adc,double S4adc,double Ecal7x7)

  {
    VMadc_ = VMadc ;
    V3adc_ = V3adc ;
    V6adc_ = V6adc ;
    VH1adc_ = VH1adc ;
    VH2adc_ = VH2adc ;
    VH3adc_ = VH3adc ;
    VH4adc_ = VH4adc ;
    CK2adc_ = CK2adc ;
    CK3adc_ = CK3adc ;
    SciVLEadc_ = SciVLEadc ;
    Sci521adc_ = Sci521adc ;
    Sci528adc_ = Sci528adc ;
    S1adc_ = S1adc ;
    S2adc_ = S2adc ;
    S3adc_ = S3adc ;
    S4adc_ = S4adc ;
    Ecal7x7_ = Ecal7x7 ;
  }
   void  HcalTBBeamCounters::setADCs06 (double VMFadc,double VMBadc,
                                     double VM1adc ,double VM2adc,double VM3adc,double VM4adc,
                                     double VM5adc ,double VM6adc,double VM7adc,double VM8adc,
                                     double CK1adc,double CK2adc,double CK3adc,
                                     double S1adc,double S2adc,double S3adc,double S4adc,
                                     double TOF1Sadc,double TOF1Jadc,double TOF2Sadc,double TOF2Jadc,
                                     double Sci521adc,double Sci528adc,
				     double BH1adc ,double BH2adc,double BH3adc,double BH4adc)
  {
   VMFadc_    = VMFadc ;
   VMBadc_    = VMBadc ;
   VM1adc_    = VM1adc ;
   VM2adc_    = VM2adc ;
   VM3adc_    = VM3adc ;
   VM4adc_    = VM4adc ;
   VM5adc_    = VM5adc ;
   VM6adc_    = VM6adc ;
   VM7adc_    = VM7adc ;
   VM8adc_    = VM8adc ;
   CK1adc_    = CK1adc ;
   CK2adc_    = CK2adc ;
   CK3adc_    = CK3adc ;
   S1adc_     = S1adc ;
   S2adc_     = S2adc ;
   S3adc_     = S3adc ;
   S4adc_     = S4adc ;
   TOF1Sadc_  = TOF1Sadc ;
   TOF1Jadc_  = TOF1Jadc ;
   TOF2Sadc_  = TOF2Sadc ;
   TOF2Jadc_  = TOF2Jadc ;
   Sci521adc_ = Sci521adc ;
   Sci528adc_ = Sci528adc ;
   BH1adc_ = BH1adc ;
   BH2adc_ = BH2adc ;
   BH3adc_ = BH3adc ;
   BH4adc_ = BH4adc ;
  }


  ostream& operator<<(ostream& s, const HcalTBBeamCounters& htbcnt) {

    s << "S1 adc     = " << htbcnt.S1adc() << endl;
    s << "S2 adc     = " << htbcnt.S2adc() << endl;
    s << "S3 adc     = " << htbcnt.S3adc() << endl;
    s << "S4 adc     = " << htbcnt.S4adc() << endl;

    return s;
  }
