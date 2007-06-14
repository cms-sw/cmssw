#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"

#include <iostream>

using  namespace std;

EcalFenixTcpFormat::EcalFenixTcpFormat(const EcalTPParameters *ecaltpp, bool tcpFormat, bool debug, bool famos,int binOfMax)
  : ecaltpp_(ecaltpp),tcpFormat_(tcpFormat),debug_(debug),famos_(famos),binOfMax_(binOfMax)
{
}
 
EcalFenixTcpFormat::~EcalFenixTcpFormat() {
}

 
void EcalFenixTcpFormat::process(std::vector<int> &Et, std::vector<int> &fgvb, int eTTotShift,
				 std::vector<EcalTriggerPrimitiveSample> & out,
				 std::vector<EcalTriggerPrimitiveSample> & out2){
  // put TP-s in the output
  // on request also in TcpFormat    
  // for famos version we have to write dummies except for the middle

  unsigned int nrSam=Et.size();
  if (famos_) nrSam=out.size();
  for (unsigned int i=0; i<nrSam;++i) {
    int myEt=0;
    int myFgvb=0;
    if (famos_ && i==binOfMax_-1) {
      myEt=Et[0];
      myFgvb=fgvb[0];
    }
    else if (!famos_) {
      myEt=Et[i];
      myFgvb=fgvb[i];
    }

    myEt=myEt>>eTTotShift;
    if (myEt>0x3ff) myEt=0x3ff ;
    int lut_out = (*lut_)[myEt] ;
    int ttFlag = (lut_out & 0x700) >> 8 ;
    if (tcpFormat_)  {
      out2[i]=EcalTriggerPrimitiveSample( ((ttFlag&0x7)<<11) | ((myFgvb & 0x1)<<10) |  (myEt & 0x3ff));
    }
    myEt = lut_out & 0xff ;
    out[i]=EcalTriggerPrimitiveSample( myEt,myFgvb,ttFlag); 
  }
}

void EcalFenixTcpFormat::setParameters(int SM, int towerInSM) 
{
    lut_ = ecaltpp_->getTowerParameters(SM, towerInSM,debug_) ;
}

