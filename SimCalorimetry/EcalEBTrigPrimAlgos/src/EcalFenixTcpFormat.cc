#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

EcalFenixTcpFormat::EcalFenixTcpFormat(bool tcpFormat, bool debug, bool famos, int binOfMax)
    : tcpFormat_(tcpFormat), debug_(debug), famos_(famos), binOfMax_(binOfMax) {
  status_ = 0;
  badTTStatus_ = &status_;
}

EcalFenixTcpFormat::~EcalFenixTcpFormat() {}

void EcalFenixTcpFormat::process(std::vector<int> &Etin, std::vector<int> &Etout) {
  // put TP-s in the output
  // on request also in TcpFormat
  // for famos version we have to write dummies except for the middle
  // std::cout << "   EcalFenixTcpFormat::process(... Etout size " << Etout.size() << "  Et size " << Etin.size() << std::endl;

  int myEt;
  int eTTotShift = 2;

  //  std::cout << " FenixTcpFormatter Etin size() " << Etin.size() << std::endl;
  for (unsigned int i = 0; i < Etin.size(); ++i) {
    // bug fix 091009:
    myEt = Etin[i];
    //std::cout << " Et " << myEt << std::endl;
    if (myEt > 0xfff)
      myEt = 0xfff;

    myEt >>= eTTotShift;
    //std::cout << " after myEt>>= eTTotShift " << myEt << std::endl;
    if (myEt > 0x3ff)
      myEt = 0x3ff;

    //myEt = lut_out & 0xff ;
    // the lut is foreseen for 8 bits. Useless to use it here

    // stay with 10 bits
    Etout[i] = myEt;
  }
}

void EcalFenixTcpFormat::process(std::vector<int> &Et,
                                 std::vector<int> &fgvb,
                                 std::vector<int> &sfgvb,
                                 int eTTotShift,
                                 std::vector<EcalEBTriggerPrimitiveSample> &out,
                                 std::vector<EcalEBTriggerPrimitiveSample> &out2,
                                 bool isInInnerRings) {
  // put TP-s in the output
  // on request also in TcpFormat
  // for famos version we have to write dummies except for the middle
  //std::cout << "   EcalFenixTcpFormat::process(... out size " << out.size() << "  Et size " << Et.size() << " Et[0] " << Et[0] << std::endl;

  int myEt;
  if (famos_) {
    for (unsigned int i = 0; i < out.size(); ++i) {
      if (i == binOfMax_ - 1) {
        myEt = Et[0] >> eTTotShift;
        if (myEt > 0x3ff)
          myEt = 0x3ff;
        if (isInInnerRings)
          myEt = myEt / 2;

        // badTTStatus_ ==0 if the TT works
        // badTTStatus_ !=0 if there are some problems
        int lut_out;
        if (*badTTStatus_ != 0) {
          lut_out = 0;
        } else
          lut_out = (lut_)[myEt];

        // int ttFlag = (lut_out & 0x700) >> 8 ;
        myEt = lut_out & 0xff;
        //	out[i]=EcalEBTriggerPrimitiveSample( myEt,fgvb[0],sfgvb[0],ttFlag);
        out[i] = EcalEBTriggerPrimitiveSample(myEt);
      } else
        out[i] = EcalEBTriggerPrimitiveSample();
    }
  } else {
    //std::cout << " FenixTcpFormatter et.size() " << Et.size() << std::endl;
    for (unsigned int i = 0; i < Et.size(); ++i) {
      //int myFgvb=fgvb[i];
      int mysFgvb = sfgvb[i];
      //myEt=Et[i]>>eTTotShift;
      //if (myEt>0x3ff) myEt=0x3ff ;
      //if (isInInnerRings) myEt = myEt /2 ;

      // bug fix 091009:
      myEt = Et[i];
      //std::cout << " Et " << myEt << std::endl;
      if (myEt > 0xfff)
        myEt = 0xfff;
      if (isInInnerRings)
        myEt = myEt / 2;
      myEt >>= eTTotShift;
      //std::cout << " after myEt>>= eTTotShift " << myEt << std::endl;
      if (myEt > 0x3ff)
        myEt = 0x3ff;

      // Spike killing
      if ((myEt > spikeZeroThresh_) && (mysFgvb == 0)) {
        myEt = 0;
      }

      int lut_out;
      if (*badTTStatus_ != 0) {
        lut_out = 0;
      } else
        lut_out = (lut_)[myEt];

      //int ttFlag = (lut_out & 0x700) >> 8 ;
      if (tcpFormat_) {
        out2[i] = EcalEBTriggerPrimitiveSample(myEt & 0x3ff);
        //std::cout << " FenixTcpFormatter final et " << (myEt & 0x3ff) << std::endl;
      }

      myEt = lut_out & 0xff;
      //std::cout << " FenixTcpFormatter final lut_out " << lut_out << " 0xff " << 0xff << " et " << myEt << std::endl;
      out[i] = EcalEBTriggerPrimitiveSample(myEt);
    }
  }
}

void EcalFenixTcpFormat::setParameters(uint32_t towid,
                                       const EcalTPGLutGroup *ecaltpgLutGroup,
                                       const EcalTPGLutIdMap *ecaltpgLut,
                                       const EcalTPGTowerStatus *ecaltpgbadTT,
                                       const EcalTPGSpike *ecaltpgSpike) {
  // Get TP zeroing threshold - defaut to 1023 for old data (no record found or EE)
  spikeZeroThresh_ = 1023;
  if (ecaltpgSpike != nullptr) {
    const EcalTPGSpike::EcalTPGSpikeMap &spikeMap = ecaltpgSpike->getMap();
    EcalTPGSpike::EcalTPGSpikeMapIterator sit = spikeMap.find(towid);
    if (sit != spikeMap.end()) {
      spikeZeroThresh_ = sit->second;
    }
  }

  const EcalTPGGroups::EcalTPGGroupsMap &groupmap = ecaltpgLutGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(towid);
  if (it != groupmap.end()) {
    uint32_t lutid = (*it).second;
    const EcalTPGLutIdMap::EcalTPGLutMap &lutmap = ecaltpgLut->getMap();
    EcalTPGLutIdMap::EcalTPGLutMapItr itl = lutmap.find(lutid);
    if (itl != lutmap.end()) {
      lut_ = (*itl).second.getLut();
      //std::cout << " FenixTcpFormatter lut_ " << std::dec<<lut_ << std::endl;
    } else
      edm::LogWarning("EcalTPG") << " could not find EcalTPGLutMap for " << lutid;

  } else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGFineGrainTowerEEMap for " << towid;

  const EcalTPGTowerStatusMap &badTTMap = ecaltpgbadTT->getMap();
  EcalTPGTowerStatusMapIterator itbadTT = badTTMap.find(towid);
  if (itbadTT != badTTMap.end()) {
    badTTStatus_ = &(*itbadTT).second;
  }
}
