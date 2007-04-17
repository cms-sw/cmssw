#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>
#include <iostream>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>

EcalFenixFgvbEB::EcalFenixFgvbEB(DBInterface * db)
  : db_(db)
{
}

EcalFenixFgvbEB::~EcalFenixFgvbEB(){
}

std::vector<int> EcalFenixFgvbEB::process( std::vector<int> add_out, std::vector<int> maxof2_out) {

    std::vector<int> output(add_out.size());
    int Elow, Ehigh, Tlow, Thigh, lut;
    int ERatLow,ERatHigh;
    std::vector<int> add_out_8(add_out.size());
    int COMP3, COMP2, COMP1, COMP0;
   
    Elow = params_[1024];
    Ehigh = params_[1025];
    Tlow = params_[1026];
    Thigh = params_[1027];
    lut = params_[1028];
    
    if (Tlow > 127) Tlow=Tlow-128;
    if (Thigh > 127) Thigh=Thigh-128;
   
    for (unsigned int i =0;i<add_out.size();i++) {
      
      ERatLow=add_out[i]*Tlow>>7;
      if (ERatLow>0xFFF) ERatLow=0xFFF;
      ERatHigh=add_out[i]*Thigh>>7;
      if (ERatHigh>0xFFF) ERatHigh=0xFFF;
      if (add_out[i] >0XFF) add_out_8[i]=0xFF; else add_out_8[i]=add_out[i];

      if( maxof2_out[i] >= ERatLow)  COMP3=1; else COMP3=0;
      if( maxof2_out[i] >= ERatHigh) COMP2=1; else COMP2=0;
      if( add_out_8[i]  >= Elow)     COMP1=1; else COMP1=0;
      if( add_out_8[i]  >= Ehigh)    COMP0=1; else COMP0=0;

      int ilut= (COMP3<<3) + (COMP2<<2) + (COMP1<<1) + COMP0;
      int mask = 1<<(ilut);
      output[i]= (lut) & (mask);
      if (output[i]>0) output[i]=1;
    }
    return output;
  }


void EcalFenixFgvbEB::setParameters(int SM, int towNum)
{
  params_ = db_->getTowerParameters(SM, towNum);
}
