#ifndef HcalSimAlgos_HcalTriggerPrimitiveAlgo_h
#define HcalSimAlgos_HcalTriggerPrimitiveAlgo_h

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureHFEMBit.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <vector>

class CaloGeometry;
class IntegerCaloSamples;

class HcalTriggerPrimitiveAlgo {
public:
  HcalTriggerPrimitiveAlgo(bool pf,
                           const std::vector<double>& w,
                           int latency,
                           uint32_t FG_threshold,
                           const std::vector<uint32_t>& FG_HF_thresholds,
                           uint32_t ZS_threshold,
                           int numberOfSamples,
                           int numberOfPresamples,
                           int numberOfFilterPresamplesHBQIE11,
                           int numberOfFilterPresamplesHEQIE11,
                           int numberOfSamplesHF,
                           int numberOfPresamplesHF,
                           bool useTDCInMinBiasBits,
                           uint32_t minSignalThreshold = 0,
                           uint32_t PMT_NoiseThreshold = 0);
  ~HcalTriggerPrimitiveAlgo();

  template <typename... Digis>
  void run(const HcalTPGCoder* incoder,
           const HcalTPGCompressor* outcoder,
           const HcalDbService* conditions,
           HcalTrigPrimDigiCollection& result,
           const HcalTrigTowerGeometry* trigTowerGeometry,
           float rctlsb,
           const HcalFeatureBit* LongvrsShortCut,
           const Digis&... digis);

  template <typename T, typename... Args>
  void addDigis(const T& collection, const Args&... digis) {
    addDigis(collection);
    addDigis(digis...);
  };

  template <typename T>
  void addDigis(const T& collection) {
    for (const auto& digi : collection) {
      addSignal(digi);
    }
  };

  template <typename D>
  void addDigis(const HcalDataFrameContainer<D>& collection) {
    for (auto i = collection.begin(); i != collection.end(); ++i) {
      D digi(*i);
      addSignal(digi);
    }
  };

  void runZS(HcalTrigPrimDigiCollection& tp);
  void runFEFormatError(const FEDRawDataCollection* rawraw,
                        const HcalElectronicsMap* emap,
                        HcalTrigPrimDigiCollection& result);
  void setPeakFinderAlgorithm(int algo);
  void setWeightsQIE11(const edm::ParameterSet& weightsQIE11);
  void setWeightQIE11(int aieta, double weight);
  void setNCTScaleShift(int);
  void setRCTScaleShift(int);

  void setNumFilterPresamplesHBQIE11(int presamples) { numberOfFilterPresamplesHBQIE11_ = presamples; }

  void setNumFilterPresamplesHEQIE11(int presamples) { numberOfFilterPresamplesHEQIE11_ = presamples; }

  void setUpgradeFlags(bool hb, bool he, bool hf);
  void setFixSaturationFlag(bool fix_saturation);
  void overrideParameters(const edm::ParameterSet& ps);

private:
  /// adds the signal to the map
  void addSignal(const HBHEDataFrame& frame);
  void addSignal(const HFDataFrame& frame);
  void addSignal(const QIE10DataFrame& frame);
  void addSignal(const QIE11DataFrame& frame);
  void addSignal(const IntegerCaloSamples& samples);
  void addFG(const HcalTrigTowerDetId& id, std::vector<bool>& msb);
  void addUpgradeFG(const HcalTrigTowerDetId& id, int depth, const std::vector<std::bitset<2>>& bits);

  bool passTDC(const QIE10DataFrame& digi, int ts) const;
  bool validUpgradeFG(const HcalTrigTowerDetId& id, int depth) const;
  bool validChannel(const QIE10DataFrame& digi, int ts) const;
  bool needLegacyFG(const HcalTrigTowerDetId& id) const;
  bool needUpgradeID(const HcalTrigTowerDetId& id, int depth) const;

  /// adds the actual digis
  void analyze(IntegerCaloSamples& samples, HcalTriggerPrimitiveDigi& result);
  // 2017 and later: QIE11
  void analyzeQIE11(IntegerCaloSamples& samples,
                    std::vector<bool> sample_saturation,
                    HcalTriggerPrimitiveDigi& result,
                    const HcalFinegrainBit& fg_algo);
  // Version 0: RCT
  void analyzeHF(IntegerCaloSamples& samples, HcalTriggerPrimitiveDigi& result, const int hf_lumi_shift);
  // Version 1: 1x1
  void analyzeHF2016(const IntegerCaloSamples& SAMPLES,
                     HcalTriggerPrimitiveDigi& result,
                     const int HF_LUMI_SHIFT,
                     const HcalFeatureBit* HCALFEM);
  // With dual anode readout
  void analyzeHFQIE10(const IntegerCaloSamples& SAMPLES,
                      HcalTriggerPrimitiveDigi& result,
                      const int HF_LUMI_SHIFT,
                      const HcalFeatureBit* HCALFEM);

  // Member initialized by constructor
  const HcaluLUTTPGCoder* incoder_;
  const HcalTPGCompressor* outcoder_;
  const HcalDbService* conditions_;
  double theThreshold;
  bool peakfind_;
  std::vector<double> weights_;
  std::array<std::array<double, 2>, 29> weightsQIE11_;
  int latency_;
  uint32_t FG_threshold_;
  std::vector<uint32_t> FG_HF_thresholds_;
  uint32_t ZS_threshold_;
  int ZS_threshold_I_;
  int numberOfSamples_;
  int numberOfPresamples_;
  int numberOfFilterPresamplesHBQIE11_;
  int numberOfFilterPresamplesHEQIE11_;
  int numberOfSamplesHF_;
  int numberOfPresamplesHF_;
  bool useTDCInMinBiasBits_;
  uint32_t minSignalThreshold_;
  uint32_t PMT_NoiseThreshold_;
  int NCTScaleShift;
  int RCTScaleShift;

  // Algo1: isPeak = TS[i-1] < TS[i] && TS[i] >= TS[i+1]
  // Algo2: isPeak = TSS[i-1] < TSS[i] && TSS[i] >= TSS[i+1],
  // TSS[i] = TS[i] + TS[i+1]
  // Default: Algo2
  int peak_finder_algorithm_;

  // Member not initialzed
  //std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & id) const;

  const HcalTrigTowerGeometry* theTrigTowerGeometry;

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;

  typedef std::map<HcalTrigTowerDetId, std::vector<bool>> SatMap;
  SatMap theSatMap;

  struct HFDetails {
    IntegerCaloSamples long_fiber;
    IntegerCaloSamples short_fiber;
    HFDataFrame ShortDigi;
    HFDataFrame LongDigi;
  };
  typedef std::map<HcalTrigTowerDetId, std::map<uint32_t, HFDetails>> HFDetailMap;
  HFDetailMap theHFDetailMap;

  struct HFUpgradeDetails {
    IntegerCaloSamples samples;
    QIE10DataFrame digi;
    std::vector<bool> validity;
    std::vector<std::bitset<2>> fgbits;
    std::vector<bool> passTDC;
  };
  typedef std::map<HcalTrigTowerDetId, std::map<uint32_t, std::array<HFUpgradeDetails, 4>>> HFUpgradeDetailMap;
  HFUpgradeDetailMap theHFUpgradeDetailMap;

  typedef std::vector<IntegerCaloSamples> SumFGContainer;
  typedef std::map<HcalTrigTowerDetId, SumFGContainer> TowerMapFGSum;
  TowerMapFGSum theTowerMapFGSum;

  // ==============================
  // =  HF Veto
  // ==============================
  // Sum = Long + Short;" // intermediate calculation.
  //  if ((Short < MinSignalThresholdET OR Long  < MinSignalThresholdET)
  //     AND Sum > PMTNoiseThresholdET) VetoedSum = 0;
  //  else VetoedSum = Sum;
  // ==============================
  // Map from FG id to veto booleans
  HcalFeatureBit* LongvrsShortCut;
  typedef std::map<uint32_t, std::vector<bool>> TowerMapVeto;
  TowerMapVeto HF_Veto;

  typedef std::map<HcalTrigTowerDetId, std::vector<bool>> FGbitMap;
  FGbitMap fgMap_;

  typedef std::vector<HcalFinegrainBit::Tower> FGUpgradeContainer;
  typedef std::map<HcalTrigTowerDetId, FGUpgradeContainer> FGUpgradeMap;
  FGUpgradeMap fgUpgradeMap_;

  bool upgrade_hb_ = false;
  bool upgrade_he_ = false;
  bool upgrade_hf_ = false;

  bool fix_saturation_ = false;

  edm::ParameterSet override_parameters_;

  bool override_adc_hf_ = false;
  uint32_t override_adc_hf_value_;
  bool override_tdc_hf_ = false;
  unsigned long long override_tdc_hf_value_;

  // HE constants
  static const int HBHE_OVERLAP_TOWER = 16;
  static const int FIRST_DEPTH7_TOWER = 26;
  static const int LAST_FINEGRAIN_DEPTH = 6;
  static const int LAST_FINEGRAIN_TOWER = 28;

  // Fine-grain in HF ignores tower 29, and starts with 30
  static const int FIRST_FINEGRAIN_TOWER = 30;

  static const int QIE8_LINEARIZATION_ET = HcaluLUTTPGCoder::QIE8_LUT_BITMASK;
  static const int QIE10_LINEARIZATION_ET = HcaluLUTTPGCoder::QIE10_LUT_BITMASK;
  static const int QIE11_LINEARIZATION_ET = HcaluLUTTPGCoder::QIE11_LUT_BITMASK;
  // Consider CaloTPGTranscoderULUT.h for values
  static const int QIE10_MAX_LINEARIZATION_ET = 0x7FF;
  static const int QIE11_MAX_LINEARIZATION_ET = 0x7FF;
};

template <typename... Digis>
void HcalTriggerPrimitiveAlgo::run(const HcalTPGCoder* incoder,
                                   const HcalTPGCompressor* outcoder,
                                   const HcalDbService* conditions,
                                   HcalTrigPrimDigiCollection& result,
                                   const HcalTrigTowerGeometry* trigTowerGeometry,
                                   float rctlsb,
                                   const HcalFeatureBit* LongvrsShortCut,
                                   const Digis&... digis) {
  theTrigTowerGeometry = trigTowerGeometry;

  incoder_ = dynamic_cast<const HcaluLUTTPGCoder*>(incoder);
  outcoder_ = outcoder;
  conditions_ = conditions;

  theSumMap.clear();
  theSatMap.clear();
  theTowerMapFGSum.clear();
  HF_Veto.clear();
  fgMap_.clear();
  fgUpgradeMap_.clear();
  theHFDetailMap.clear();
  theHFUpgradeDetailMap.clear();

  // Add all digi collections
  addDigis(digis...);

  // Prepare the fine-grain calculation algorithm for HB/HE
  int version = 0;
  if (upgrade_he_ or upgrade_hb_)
    version = conditions_->getHcalTPParameters()->getFGVersionHBHE();
  if (override_parameters_.exists("FGVersionHBHE"))
    version = override_parameters_.getParameter<uint32_t>("FGVersionHBHE");
  HcalFinegrainBit fg_algo(version);

  // VME produces additional bits on the front used by lumi but not the
  // trigger, this shift corrects those out by right shifting over them.
  for (auto& item : theSumMap) {
    result.push_back(HcalTriggerPrimitiveDigi(item.first));
    HcalTrigTowerDetId detId(item.second.id());
    if (detId.ietaAbs() >= theTrigTowerGeometry->firstHFTower(detId.version())) {
      if (detId.version() == 0) {
        analyzeHF(item.second, result.back(), RCTScaleShift);
      } else if (detId.version() == 1) {
        if (upgrade_hf_)
          analyzeHFQIE10(item.second, result.back(), NCTScaleShift, LongvrsShortCut);
        else
          analyzeHF2016(item.second, result.back(), NCTScaleShift, LongvrsShortCut);
      } else {
        // Things are going to go poorly
      }
    } else {
      // Determine which energy reconstruction path to take based on the
      // fine-grain availability:
      //  * QIE8 TP add entries into fgMap_
      //  * QIE11 TP add entries into fgUpgradeMap_
      //    (not for tower 16 unless HB is upgraded, too)
      if (fgMap_.find(item.first) != fgMap_.end()) {
        analyze(item.second, result.back());
      } else if (fgUpgradeMap_.find(item.first) != fgUpgradeMap_.end()) {
        SatMap::iterator item_sat = theSatMap.find(detId);
        if (item_sat == theSatMap.end())
          analyzeQIE11(item.second, std::vector<bool>(), result.back(), fg_algo);
        else
          analyzeQIE11(item.second, item_sat->second, result.back(), fg_algo);
      }
    }
  }

  // Free up some memory
  theSumMap.clear();
  theSatMap.clear();
  theTowerMapFGSum.clear();
  HF_Veto.clear();
  fgMap_.clear();
  fgUpgradeMap_.clear();
  theHFDetailMap.clear();
  theHFUpgradeDetailMap.clear();

  return;
}

#endif
