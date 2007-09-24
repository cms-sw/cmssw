#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using  namespace std;

EcalFenixTcpFormat::EcalFenixTcpFormat(bool tcpFormat, bool debug, bool famos,int binOfMax)
  : tcpFormat_(tcpFormat),debug_(debug),famos_(famos),binOfMax_(binOfMax)
{
}
 
EcalFenixTcpFormat::~EcalFenixTcpFormat() {
}

 
void EcalFenixTcpFormat::process(std::vector<int> &Et, std::vector<int> &fgvb, int eTTotShift,
				 std::vector<EcalTriggerPrimitiveSample> & out,
				 std::vector<EcalTriggerPrimitiveSample> & out2, bool isInInnerRings){
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
    int lut_out = (lut_)[myEt] ;
    int ttFlag = (lut_out & 0x700) >> 8 ;
    if (tcpFormat_)  {
      out2[i]=EcalTriggerPrimitiveSample( ((ttFlag&0x7)<<11) | ((myFgvb & 0x1)<<10) |  (myEt & 0x3ff));
    }
    myEt = lut_out & 0xff ;
    out[i]=EcalTriggerPrimitiveSample( myEt,myFgvb,ttFlag); 
  }
}

void EcalFenixTcpFormat::setParameters(uint32_t towid,const EcalTPGLutGroup *ecaltpgLutGroup, const EcalTPGLutIdMap *ecaltpgLut)
{
  const EcalTPGGroups::EcalTPGGroupsMap & groupmap = ecaltpgLutGroup -> getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it=groupmap.find(towid);
  if (it!=groupmap.end()) {
    uint32_t lutid=(*it).second;
    const EcalTPGLutIdMap::EcalTPGLutMap &lutmap = ecaltpgLut-> getMap();
    EcalTPGLutIdMap::EcalTPGLutMapItr itl=lutmap.find(lutid);
    if (itl!=lutmap.end()) {
      lut_=(*itl).second.getLut();
    }  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGLutMap for "<<lutid;

  }
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGFineGrainTowerEEMap for "<<towid;
}

