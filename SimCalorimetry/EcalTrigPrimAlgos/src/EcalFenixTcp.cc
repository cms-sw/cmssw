#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h>

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include <CondFormats/EcalObjects/interface/EcalTPGTPMode.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
//----------------------------------------------------------------------------------------
EcalFenixTcp::EcalFenixTcp(
    bool tcpFormat, bool debug, bool famos, int binOfMax, int maxNrSamples, int nbMaxStrips, bool tpInfoPrintout)
    : debug_(debug), nbMaxStrips_(nbMaxStrips), tpInfoPrintout_(tpInfoPrintout) {
  bypasslin_.resize(nbMaxStrips_);
  for (int i = 0; i < nbMaxStrips_; i++)
    bypasslin_[i] = new EcalFenixBypassLin();
  adder_ = new EcalFenixEtTot();
  maxOf2_ = new EcalFenixMaxof2(maxNrSamples, nbMaxStrips_);
  formatter_EB_ = new EcalFenixTcpFormatEB(tcpFormat, debug_, famos, binOfMax);
  formatter_EE_ = new EcalFenixTcpFormatEE(tcpFormat, debug_, famos, binOfMax);
  fgvbEB_ = new EcalFenixFgvbEB(maxNrSamples);
  fgvbEE_ = new EcalFenixTcpFgvbEE(maxNrSamples);
  sfgvbEB_ = new EcalFenixTcpsFgvbEB();

  // permanent data structures
  bypasslin_out_.resize(nbMaxStrips_);
  std::vector<int> vec(maxNrSamples, 0);
  for (int i = 0; i < nbMaxStrips_; i++)
    bypasslin_out_[i] = vec;

  adder_even_out_.resize(maxNrSamples);
  adder_odd_out_.resize(maxNrSamples);
  maxOf2_out_.resize(maxNrSamples);
  fgvb_out_.resize(maxNrSamples);
  strip_fgvb_out_.resize(maxNrSamples);
}
//-----------------------------------------------------------------------------------------
EcalFenixTcp::~EcalFenixTcp() {
  for (int i = 0; i < nbMaxStrips_; i++)
    delete bypasslin_[i];
  delete adder_;
  delete maxOf2_;
  delete formatter_EB_;
  delete formatter_EE_;
  delete fgvbEB_;
  delete fgvbEE_;
}
//-----------------------------------------------------------------------------------------

void EcalFenixTcp::process(std::vector<EBDataFrame> &bid,  // dummy argument for template call
                           std::vector<std::vector<int>> &tpframetow,
                           int nStr,
                           std::vector<EcalTriggerPrimitiveSample> &tptow,
                           std::vector<EcalTriggerPrimitiveSample> &tptow2,
                           bool isInInnerRing,
                           EcalTrigTowerDetId towid) {
  int bitMask = 12;
  // The 14th bit is always used for the odd>even flag. If the flagging is off in the Strip fenix the feature will be not used.
  int bitOddEven = 13;
  process_part1(tpframetow, nStr, bitMask, bitOddEven);

  process_part2_barrel(tpframetow,
                       nStr,
                       bitMask,
                       bitOddEven,
                       ecaltpgFgEBGroup_,
                       ecaltpgLutGroup_,
                       ecaltpgLut_,
                       ecaltpgFineGrainEB_,
                       ecaltpgBadTT_,
                       ecaltpgSpike_,
                       tptow,
                       tptow2,
                       towid);
}

//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process(std::vector<EEDataFrame> &bid,  // dummy argument for template call
                           std::vector<std::vector<int>> &tpframetow,
                           int nStr,
                           std::vector<EcalTriggerPrimitiveSample> &tptow,
                           std::vector<EcalTriggerPrimitiveSample> &tptow2,
                           bool isInInnerRing,
                           EcalTrigTowerDetId towid) {
  int bitMask = 12;  // Pascal: endcap has 12 bits as in EB (bug in FENIX!!!!)
  // The 14th bit is always used for the odd>even flag. If the flagging is off in the Strip fenix the feature will be not used.
  int bitOddEven = 13;

  process_part1(tpframetow, nStr, bitMask, bitOddEven);

  process_part2_endcap(tpframetow,
                       nStr,
                       bitMask,
                       bitOddEven,
                       ecaltpgLutGroup_,
                       ecaltpgLut_,
                       ecaltpgFineGrainTowerEE_,
                       ecaltpgBadTT_,
                       tptow,
                       tptow2,
                       isInInnerRing,
                       towid);
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part1(std::vector<std::vector<int>> &tpframetow, int nStr, int bitMask, int bitOddEven) {
  // call adder
  this->getAdder()->process(tpframetow, nStr, bitMask, bitOddEven, adder_even_out_, adder_odd_out_);
  if (debug_) {
    edm::LogVerbatim("EcalTPG") << "output of TCP adder is a vector of size: " << adder_even_out_.size();
    edm::LogVerbatim("EcalTPG") << "EVEN sum : ";
    std::string even_adder_outputs;
    for (unsigned int i = 0; i < adder_even_out_.size(); i++) {
      even_adder_outputs.append(" ");
      even_adder_outputs.append(std::to_string(adder_even_out_[i]));
    }
    edm::LogVerbatim("EcalTPG") << even_adder_outputs << "\n";

    edm::LogVerbatim("EcalTPG") << "ODD sum : ";
    std::string odd_adder_outputs;
    for (unsigned int i = 0; i < adder_odd_out_.size(); i++) {
      odd_adder_outputs.append(" ");
      odd_adder_outputs.append(std::to_string(adder_odd_out_[i]));
    }
    edm::LogVerbatim("EcalTPG") << odd_adder_outputs << "\n";
  }
  return;
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_barrel(std::vector<std::vector<int>> &bypasslinout,
                                        int nStr,
                                        int bitMask,
                                        int bitOddEven,
                                        const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                                        const EcalTPGLutGroup *ecaltpgLutGroup,
                                        const EcalTPGLutIdMap *ecaltpgLut,
                                        const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
                                        const EcalTPGTowerStatus *ecaltpgBadTT,
                                        const EcalTPGSpike *ecaltpgSpike,
                                        std::vector<EcalTriggerPrimitiveSample> &tcp_out,
                                        std::vector<EcalTriggerPrimitiveSample> &tcp_outTcc,
                                        EcalTrigTowerDetId towid) {
  // call maxof2
  // the oddEven flag is used to exclude "odd" strip from the computation of the maxof2 as in the fenix firmware
  this->getMaxOf2()->process(bypasslinout, nStr, bitMask, bitOddEven, maxOf2_out_);

  if (debug_) {
    edm::LogVerbatim("EcalTPG") << "output of maxof2 is a vector of size: " << maxOf2_out_.size();
    edm::LogVerbatim("EcalTPG") << "value : ";
    std::string maxOf2_outputs;
    for (unsigned int i = 0; i < maxOf2_out_.size(); i++) {
      maxOf2_outputs.append(" ");
      maxOf2_outputs.append(std::to_string(maxOf2_out_[i]));
    }
    edm::LogVerbatim("EcalTPG") << maxOf2_outputs << "\n";
  }

  // call fgvb
  this->getFGVBEB()->setParameters(towid.rawId(), ecaltpgFgEBGroup, ecaltpgFineGrainEB);
  // The FGVB is computed only on the even sum, as in the firmware
  this->getFGVBEB()->process(adder_even_out_, maxOf2_out_, fgvb_out_);

  // Call sFGVB
  this->getsFGVBEB()->process(bypasslinout, nStr, bitMask, strip_fgvb_out_);

  if (debug_) {
    edm::LogVerbatim("EcalTPG") << "output of fgvb is a vector of size: " << fgvb_out_.size();
    edm::LogVerbatim("EcalTPG") << "value : ";
    std::string fgvb_output;
    for (unsigned int i = 0; i < fgvb_out_.size(); i++) {
      fgvb_output.append(" ");
      fgvb_output.append(std::to_string(fgvb_out_[i]));
    }
    edm::LogVerbatim("EcalTPG") << fgvb_output;
  }

  // call formatter
  int eTTotShift = 2;

  this->getFormatterEB()->setParameters(
      towid.rawId(), ecaltpgLutGroup, ecaltpgLut, ecaltpgBadTT, ecaltpgSpike, ecaltpgTPMode_);
  this->getFormatterEB()->process(
      adder_even_out_, adder_odd_out_, fgvb_out_, strip_fgvb_out_, eTTotShift, tcp_out, tcp_outTcc);

  if (tpInfoPrintout_) {
    for (unsigned int i = 3; i < tcp_out.size(); i++) {
      edm::LogVerbatim("EcalTPG") << " " << i << " " << tcp_out[i];
    }
  }

  if (debug_) {
    edm::LogVerbatim("EcalTPG") << "\noutput of TCP formatter Barrel is a vector of size: " << tcp_out.size();
    edm::LogVerbatim("EcalTPG") << "value : ";
    for (unsigned int i = 0; i < tcp_out.size(); i++) {
      edm::LogVerbatim("EcalTPG") << " " << i << " " << tcp_out[i];
    }
    edm::LogVerbatim("EcalTPG");
  }

  return;
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_endcap(std::vector<std::vector<int>> &bypasslinout,
                                        int nStr,
                                        int bitMask,
                                        int bitOddEven,
                                        const EcalTPGLutGroup *ecaltpgLutGroup,
                                        const EcalTPGLutIdMap *ecaltpgLut,
                                        const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                                        const EcalTPGTowerStatus *ecaltpgbadTT,
                                        std::vector<EcalTriggerPrimitiveSample> &tcp_out,
                                        std::vector<EcalTriggerPrimitiveSample> &tcp_outTcc,
                                        bool isInInnerRings,
                                        EcalTrigTowerDetId towid)

{
  // Zero EB strip records
  for (unsigned int i = 0; i < strip_fgvb_out_.size(); ++i) {
    strip_fgvb_out_[i] = 0;
  }

  // call fgvb
  this->getFGVBEE()->setParameters(towid.rawId(), ecaltpgFineGrainTowerEE);
  //  fgvbEE_->process(bypasslin_out_,nStr,bitMask,fgvb_out_);
  fgvbEE_->process(bypasslinout, nStr, bitMask, fgvb_out_);

  // call formatter
  int eTTotShift = 2;  // Pascal: endcap has 12 bits as in EB (bug in FENIX!!!!)
                       // so shift must be applied to just keep [11:2]

  this->getFormatterEE()->setParameters(
      towid.rawId(), ecaltpgLutGroup, ecaltpgLut, ecaltpgbadTT, nullptr, ecaltpgTPMode_);

  // Pass both the even and the odd Et sums to the EE formatter also if there is not TCP in the electronics.
  // The feature can be implemented in the TCC in the future: the emulator is kept generic.
  this->getFormatterEE()->process(
      adder_even_out_, adder_odd_out_, fgvb_out_, strip_fgvb_out_, eTTotShift, tcp_out, tcp_outTcc, isInInnerRings);
  if (debug_) {
    edm::LogVerbatim("EcalTPG") << "\noutput of TCP formatter(endcap) is a vector of size: " << tcp_out.size();
    edm::LogVerbatim("EcalTPG") << "value : ";
    for (unsigned int i = 0; i < tcp_out.size(); i++) {
      edm::LogVerbatim("EcalTPG") << " " << i << " " << tcp_out[i];
    }
    edm::LogVerbatim("EcalTPG");
  }
  return;
}
