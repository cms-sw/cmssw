#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using  namespace std;

EcalFenixTcpFormat::EcalFenixTcpFormat(bool tcpFormat, bool debug, bool famos,int binOfMax)
  : tcpFormat_(tcpFormat),debug_(debug),famos_(famos),binOfMax_(binOfMax)
{
  status_=0;
  badTTStatus_=&status_;
}
 
EcalFenixTcpFormat::~EcalFenixTcpFormat() {
}

 
void EcalFenixTcpFormat::process(std::vector<int> &Et, std::vector<int> &fgvb, int eTTotShift,
				 std::vector<EcalTriggerPrimitiveSample> & out,
				 std::vector<EcalTriggerPrimitiveSample> & out2, bool isInInnerRings){
  // put TP-s in the output
  // on request also in TcpFormat    
  // for famos version we have to write dummies except for the middle
  
  int myEt;
  if (famos_) {
    for (unsigned int i=0; i<out.size();++i) {
      if (i==binOfMax_-1) {
	myEt=Et[0]>>eTTotShift;
	if (myEt>0x3ff) myEt=0x3ff ;
	if (isInInnerRings) myEt = myEt /2 ;
	
	// badTTStatus_ ==0 if the TT works
	// badTTStatus_ !=0 if there are some problems
	int lut_out;
	if (*badTTStatus_!=0){
	  lut_out = 0;
	}
	else
	  lut_out = (lut_)[myEt] ;
	
	int ttFlag = (lut_out & 0x700) >> 8 ;
	myEt = lut_out & 0xff ;
	out[i]=EcalTriggerPrimitiveSample( myEt,fgvb[0],ttFlag); 
      }
      else out[i]=EcalTriggerPrimitiveSample( );
    }
  }
  else {
    for (unsigned int i=0; i<Et.size();++i) {
      int myFgvb=fgvb[i];
      //myEt=Et[i]>>eTTotShift;
      //if (myEt>0x3ff) myEt=0x3ff ;
      //if (isInInnerRings) myEt = myEt /2 ;  

      // bug fix 091009:
      myEt=Et[i]; 
      if (myEt>0xfff) 
	myEt=0xfff ;
      if (isInInnerRings) 
	myEt = myEt /2 ;  
      myEt >>= eTTotShift ;
      if (myEt>0x3ff) myEt=0x3ff ;

	int lut_out;
	if (*badTTStatus_!=0){
	  lut_out = 0;
	}
	else
	  lut_out = (lut_)[myEt] ;
      
      int ttFlag = (lut_out & 0x700) >> 8 ;
      if (tcpFormat_)  {
	out2[i]=EcalTriggerPrimitiveSample( ((ttFlag&0x7)<<11) | ((myFgvb & 0x1)<<10) |  (myEt & 0x3ff));
      }
      myEt = lut_out & 0xff ;
      out[i]=EcalTriggerPrimitiveSample( myEt,myFgvb,ttFlag); 
    }
  }
}

void EcalFenixTcpFormat::setParameters(uint32_t towid,const EcalTPGLutGroup *ecaltpgLutGroup, const EcalTPGLutIdMap *ecaltpgLut, const EcalTPGTowerStatus *ecaltpgbadTT)
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
  
  const EcalTPGTowerStatusMap & badTTMap = ecaltpgbadTT -> getMap();
  EcalTPGTowerStatusMapIterator itbadTT = badTTMap.find(towid);
  if (itbadTT!=badTTMap.end()) {
    badTTStatus_=&(*itbadTT).second;
  }
}

