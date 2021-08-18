#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCP_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXTCP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFgvbEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormatEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormatEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpsFgvbEB.h>

#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include <iostream>
#include <vector>

class EcalTPGFineGrainEBGroup;
class EcalTPGLutGroup;
class EcalTPGLutIdMap;
class EcalTPGFineGrainEBIdMap;
class EcalTPGFineGrainTowerEE;
class EcalTrigTowerDetId;
class EcalTPGTowerStatus;
class EcalTPGTPMode;

/**
    \class EcalFenixTcp
    \brief class representing the Fenix chip, format strip
*/
class EcalFenixTcp {
private:
  bool debug_;
  int nbMaxStrips_;
  bool tpInfoPrintout_;

  EcalFenixMaxof2 *maxOf2_;
  std::vector<EcalFenixBypassLin *> bypasslin_;
  EcalFenixEtTot *adder_;
  EcalFenixFgvbEB *fgvbEB_;
  EcalFenixTcpFgvbEE *fgvbEE_;
  EcalFenixTcpsFgvbEB *sfgvbEB_;

  EcalFenixTcpFormatEB *formatter_EB_;
  EcalFenixTcpFormatEE *formatter_EE_;

  // permanent data structures
  std::vector<std::vector<int>> bypasslin_out_;
  std::vector<int> adder_even_out_;
  std::vector<int> adder_odd_out_;
  std::vector<int> maxOf2_out_;
  std::vector<int> fgvb_out_;
  std::vector<int> strip_fgvb_out_;

public:
  // temporary, for timing tests
  void setPointers(const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                   const EcalTPGLutGroup *ecaltpgLutGroup,
                   const EcalTPGLutIdMap *ecaltpgLut,
                   const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
                   const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                   const EcalTPGTowerStatus *ecaltpgBadTT,
                   const EcalTPGSpike *ecaltpgSpike,
                   const EcalTPGTPMode *ecaltpgTPMode) {
    ecaltpgFgEBGroup_ = ecaltpgFgEBGroup;
    ecaltpgLutGroup_ = ecaltpgLutGroup;
    ecaltpgLut_ = ecaltpgLut;
    ecaltpgFineGrainEB_ = ecaltpgFineGrainEB;
    ecaltpgFineGrainTowerEE_ = ecaltpgFineGrainTowerEE;
    ecaltpgBadTT_ = ecaltpgBadTT;
    ecaltpgSpike_ = ecaltpgSpike;
    ecaltpgTPMode_ = ecaltpgTPMode;
  }
  // end temporary, for timing tests

  EcalFenixTcp(
      bool tcpFormat, bool debug, bool famos, int binOfMax, int maxNrSamples, int nbMaxStrips, bool TPinfoPrintout);
  virtual ~EcalFenixTcp();

  void process(std::vector<EBDataFrame> &bid,  // dummy argument for template call
               std::vector<std::vector<int>> &tpframetow,
               int nStr,
               std::vector<EcalTriggerPrimitiveSample> &tptow,
               std::vector<EcalTriggerPrimitiveSample> &tptow2,
               bool isInInnerRings,
               EcalTrigTowerDetId thisTower);
  void process(std::vector<EEDataFrame> &bid,  // dummy argument for template call
               std::vector<std::vector<int>> &tpframetow,
               int nStr,
               std::vector<EcalTriggerPrimitiveSample> &tptow,
               std::vector<EcalTriggerPrimitiveSample> &tptow2,
               bool isInInnerRings,
               EcalTrigTowerDetId thisTower);

  void process_part1(std::vector<std::vector<int>> &tpframetow, int nStr, int bitMask, int bitOddEven);

  void process_part2_barrel(std::vector<std::vector<int>> &,
                            int nStr,
                            int bitMask,
                            int bitOddEven,
                            const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                            const EcalTPGLutGroup *ecaltpgLutGroup,
                            const EcalTPGLutIdMap *ecaltpgLut,
                            const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
                            const EcalTPGTowerStatus *ecaltpgBadTT,
                            const EcalTPGSpike *ecaltpgSpike,
                            std::vector<EcalTriggerPrimitiveSample> &tptow,
                            std::vector<EcalTriggerPrimitiveSample> &tptow2,
                            EcalTrigTowerDetId towid);

  void process_part2_endcap(std::vector<std::vector<int>> &,
                            int nStr,
                            int bitMask,
                            int bitOddEven,
                            const EcalTPGLutGroup *ecaltpgLutGroup,
                            const EcalTPGLutIdMap *ecaltpgLut,
                            const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                            const EcalTPGTowerStatus *ecaltpgBadTT,
                            std::vector<EcalTriggerPrimitiveSample> &tptow,
                            std::vector<EcalTriggerPrimitiveSample> &tptow2,
                            bool isInInnerRings,
                            EcalTrigTowerDetId towid);

  EcalFenixBypassLin *getBypasslin(int i) const { return bypasslin_[i]; }
  EcalFenixEtTot *getAdder() const { return adder_; }
  EcalFenixMaxof2 *getMaxOf2() const { return maxOf2_; }
  EcalFenixTcpFormatEB *getFormatterEB() const { return formatter_EB_; }
  EcalFenixTcpFormatEE *getFormatterEE() const { return formatter_EE_; }
  EcalFenixFgvbEB *getFGVBEB() const { return fgvbEB_; }
  EcalFenixTcpFgvbEE *getFGVBEE() const { return fgvbEE_; }
  EcalFenixTcpsFgvbEB *getsFGVBEB() const { return sfgvbEB_; }

  const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup_;
  const EcalTPGLutGroup *ecaltpgLutGroup_;
  const EcalTPGLutIdMap *ecaltpgLut_;
  const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB_;
  const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE_;
  const EcalTPGTowerStatus *ecaltpgBadTT_;
  const EcalTPGSpike *ecaltpgSpike_;
  const EcalTPGTPMode *ecaltpgTPMode_;
};

#endif
