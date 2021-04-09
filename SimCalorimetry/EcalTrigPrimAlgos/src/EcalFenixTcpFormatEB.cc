#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTPMode.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormatEB.h>
#include <iostream>
using namespace std;

EcalFenixTcpFormatEB::EcalFenixTcpFormatEB(bool tcpFormat, bool debug, bool famos, int binOfMax)
    : tcpFormat_(tcpFormat), debug_(debug), famos_(famos), binOfMax_(binOfMax) {
  status_ = 0;
  badTTStatus_ = &status_;
}

EcalFenixTcpFormatEB::~EcalFenixTcpFormatEB() {}

void EcalFenixTcpFormatEB::process(std::vector<int> &Et_even_sum,
                                   std::vector<int> &Et_odd_sum,
                                   std::vector<int> &fgvb,
                                   std::vector<int> &sfgvb,
                                   int eTTotShift,
                                   std::vector<EcalTriggerPrimitiveSample> &out,
                                   std::vector<EcalTriggerPrimitiveSample> &out2) {
  // put TP-s in the output
  // on request also in TcpFormat
  // for famos version we have to write dummies except for the middle

  int myEt = 0;
  if (famos_) {
    for (unsigned int i = 0; i < out.size(); ++i) {
      if (i == binOfMax_ - 1) {
        myEt = Et_even_sum[i] >> eTTotShift;
        if (myEt > 0x3ff)
          myEt = 0x3ff;

        // badTTStatus_ ==0 if the TT works
        // badTTStatus_ !=0 if there are some problems
        int lut_out;
        if (*badTTStatus_ != 0) {
          lut_out = 0;
        } else
          lut_out = (lut_)[myEt];

        int ttFlag = (lut_out & 0x700) >> 8;
        myEt = lut_out & 0xff;
        out[i] = EcalTriggerPrimitiveSample(myEt, fgvb[0], sfgvb[0], ttFlag);
      } else
        out[i] = EcalTriggerPrimitiveSample();
    }
  } else {
    for (unsigned int i = 0; i < Et_even_sum.size(); ++i) {
      int myFgvb = fgvb[i];
      int mysFgvb = sfgvb[i];
      bool is_odd_larger = false;

      // Check if odd sum is larger than even sum, in case flag_EB_odd_even_tcp is used
      if (Et_odd_sum[i] > Et_even_sum[i]) {
        is_odd_larger = true;
      }

      switch (ecaltpgTPMode_->EBFenixTcpOutput) {
        case 0:  //output even sum
          myEt = Et_even_sum[i];
          break;
        case 1:  // output larger of odd and even
          if (Et_odd_sum[i] > Et_even_sum[i]) {
            myEt = Et_odd_sum[i];
          } else {
            myEt = Et_even_sum[i];
          }
          break;
        case 2:  // output even+odd
          myEt = Et_even_sum[i] + Et_odd_sum[i];
          break;
        default:
          // In case of unknown configuration switch to default
          myEt = Et_even_sum[i];
          break;
      }

      // check TPmode config to decide to output the FGVB or the odd>even flag
      int infobit1 = myFgvb;
      if (ecaltpgTPMode_->EBFenixTcpInfobit1)
        infobit1 = is_odd_larger;

      if (myEt > 0xfff)
        myEt = 0xfff;
      myEt >>= eTTotShift;
      if (myEt > 0x3ff)
        myEt = 0x3ff;

      // Spike killer
      if ((myEt > spikeZeroThresh_) && (mysFgvb == 0)) {
        myEt = 0;
      }

      int lut_out;
      if (*badTTStatus_ != 0) {
        lut_out = 0;
      } else
        lut_out = (lut_)[myEt];

      int ttFlag = (lut_out & 0x700) >> 8;
      if (tcpFormat_) {
        out2[i] = EcalTriggerPrimitiveSample(((ttFlag & 0x7) << 11) | ((infobit1 & 0x1) << 10) | (myEt & 0x3ff));
      }
      myEt = lut_out & 0xff;
      out[i] = EcalTriggerPrimitiveSample(myEt, infobit1, mysFgvb, ttFlag);
    }
  }
}

void EcalFenixTcpFormatEB::setParameters(uint32_t towid,
                                         const EcalTPGLutGroup *ecaltpgLutGroup,
                                         const EcalTPGLutIdMap *ecaltpgLut,
                                         const EcalTPGTowerStatus *ecaltpgbadTT,
                                         const EcalTPGSpike *ecaltpgSpike,
                                         const EcalTPGTPMode *ecaltpgTPMode) {
  // Get TP zeroing threshold - defaut to 1023 for old data (no record found or
  // EE)
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
    } else
      edm::LogWarning("EcalTPG") << " could not find EcalTPGLutMap for " << lutid;

  } else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGFineGrainTowerEEMap for " << towid;

  const EcalTPGTowerStatusMap &badTTMap = ecaltpgbadTT->getMap();
  EcalTPGTowerStatusMapIterator itbadTT = badTTMap.find(towid);
  if (itbadTT != badTTMap.end()) {
    badTTStatus_ = &(*itbadTT).second;
  }

  ecaltpgTPMode_ = ecaltpgTPMode;
}
