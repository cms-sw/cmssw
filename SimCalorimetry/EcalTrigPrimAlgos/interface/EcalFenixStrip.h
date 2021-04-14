#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXSTRIP_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXSTRIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixOddAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>

#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>
#include <string>

class EBDataFrame;
class EcalTriggerPrimitiveSample;
class EcalTPGSlidingWindow;
class EcalTPGFineGrainStripEE;
class EcalFenixStripFgvbEE;
class EcalFenixStripFormatEB;
class EcalFenixStripFormatEE;
class EcalTPGStripStatus;
class EcalTPGTPMode;

/**
    \class EcalFenixStrip
    \brief class representing the Fenix chip, format strip
*/
class EcalFenixStrip {
public:
  // constructor, destructor
  EcalFenixStrip(const EcalElectronicsMapping *theMapping,
                 bool debug,
                 bool famos,
                 int maxNrSamples,
                 int nbMaxXtals,
                 bool TPinfoPrintout);
  virtual ~EcalFenixStrip();

private:
  const EcalElectronicsMapping *theMapping_;

  bool debug_;
  bool famos_;
  int nbMaxXtals_;
  bool tpInfoPrintout_;

  std::vector<EcalFenixLinearizer *> linearizer_;

  EcalFenixAmplitudeFilter *amplitude_filter_;
  EcalFenixOddAmplitudeFilter *oddAmplitude_filter_;

  EcalFenixPeakFinder *peak_finder_;

  EcalFenixStripFormatEB *fenixFormatterEB_;

  EcalFenixStripFormatEE *fenixFormatterEE_;

  EcalFenixEtStrip *adder_;

  EcalFenixStripFgvbEE *fgvbEE_;

  // data formats for each event
  std::vector<std::vector<int>> lin_out_;
  std::vector<int> add_out_;

  // Data formats for even filter. Up to the adder only 1 path, then splitted.
  std::vector<int> even_filt_out_;
  std::vector<int> even_peak_out_;

  // Data formats for odd filter, as data path is duplicated for odd filter
  std::vector<int> odd_filt_out_;
  std::vector<int> odd_peak_out_;

  std::vector<int> format_out_;
  std::vector<int> fgvb_out_;
  std::vector<int> fgvb_out_temp_;

  const EcalTPGPedestals *ecaltpPed_;
  const EcalTPGLinearizationConst *ecaltpLin_;
  const EcalTPGWeightIdMap *ecaltpgWeightMap_;
  const EcalTPGWeightGroup *ecaltpgWeightGroup_;
  const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap_;
  const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup_;
  const EcalTPGSlidingWindow *ecaltpgSlidW_;
  const EcalTPGFineGrainStripEE *ecaltpgFgStripEE_;
  const EcalTPGCrystalStatus *ecaltpgBadX_;
  const EcalTPGStripStatus *ecaltpgStripStatus_;
  const EcalTPGTPMode *ecaltpgTPMode_;

  bool identif_;

public:
  void setPointers(const EcalTPGPedestals *ecaltpPed,
                   const EcalTPGLinearizationConst *ecaltpLin,
                   const EcalTPGWeightIdMap *ecaltpgWeightMap,
                   const EcalTPGWeightGroup *ecaltpgWeightGroup,
                   const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap,
                   const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup,
                   const EcalTPGSlidingWindow *ecaltpgSlidW,
                   const EcalTPGFineGrainStripEE *ecaltpgFgStripEE,
                   const EcalTPGCrystalStatus *ecaltpgBadX,
                   const EcalTPGStripStatus *ecaltpgStripStatus,
                   const EcalTPGTPMode *ecaltpgTPMode) {
    ecaltpPed_ = ecaltpPed;
    ecaltpLin_ = ecaltpLin;
    ecaltpgWeightMap_ = ecaltpgWeightMap;
    ecaltpgWeightGroup_ = ecaltpgWeightGroup;
    ecaltpgOddWeightMap_ = ecaltpgOddWeightMap;
    ecaltpgOddWeightGroup_ = ecaltpgOddWeightGroup;
    ecaltpgSlidW_ = ecaltpgSlidW;
    ecaltpgFgStripEE_ = ecaltpgFgStripEE;
    ecaltpgBadX_ = ecaltpgBadX;
    ecaltpgStripStatus_ = ecaltpgStripStatus;
    ecaltpgTPMode_ = ecaltpgTPMode;
  }

  // main methods
  void process(std::vector<EBDataFrame> &samples, int nrXtals, std::vector<int> &out);
  void process(std::vector<EEDataFrame> &samples, int nrXtals, std::vector<int> &out);

  template <class T>
  void process_part1(int identif,
                     std::vector<T> &df,
                     int nrXtals,
                     uint32_t stripid,
                     const EcalTPGPedestals *ecaltpPed,
                     const EcalTPGLinearizationConst *ecaltpLin,
                     const EcalTPGWeightIdMap *ecaltpgWeightMap,
                     const EcalTPGWeightGroup *ecaltpgWeightGroup,
                     const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap,
                     const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup,
                     const EcalTPGCrystalStatus *ecaltpBadX);
  // process method is splitted in 2 parts:
  //   the first one is overloaded, the same except input
  //   the second part is slightly different for barrel/endcap
  void process_part2_barrel(uint32_t stripid,
                            const EcalTPGSlidingWindow *ecaltpgSlidW,
                            const EcalTPGFineGrainStripEE *ecaltpgFgStripEE);

  void process_part2_endcap(uint32_t stripid,
                            const EcalTPGSlidingWindow *ecaltpgSlidW,
                            const EcalTPGFineGrainStripEE *ecaltpgFgStripEE,
                            const EcalTPGStripStatus *ecaltpgStripStatus);

  // getters for the algorithms  ;

  EcalFenixLinearizer *getLinearizer(int i) const { return linearizer_[i]; }
  EcalFenixEtStrip *getAdder() const { return adder_; }
  EcalFenixAmplitudeFilter *getEvenFilter() const { return amplitude_filter_; }
  EcalFenixOddAmplitudeFilter *getOddFilter() const { return oddAmplitude_filter_; }
  EcalFenixPeakFinder *getPeakFinder() const { return peak_finder_; }

  EcalFenixStripFormatEB *getFormatterEB() const { return fenixFormatterEB_; }
  EcalFenixStripFormatEE *getFormatterEE() const { return fenixFormatterEE_; }

  EcalFenixStripFgvbEE *getFGVB() const { return fgvbEE_; }

  void setbadStripMissing(bool flag) { identif_ = flag; }
  bool getbadStripMissing() const { return identif_; }
};
#endif
