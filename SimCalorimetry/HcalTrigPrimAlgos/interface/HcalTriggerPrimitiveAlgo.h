#ifndef HcalSimAlgos_HcalTriggerPrimitiveAlgo_h
#define HcalSimAlgos_HcalTriggerPrimitiveAlgo_h

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureHFEMBit.h"//cuts based on short and long energy deposited.

#include <map>
#include <vector>
class CaloGeometry;
class IntegerCaloSamples;

class HcalTriggerPrimitiveAlgo {
public:
  HcalTriggerPrimitiveAlgo(bool pf, const std::vector<double>& w, 
                           int latency,
                           bool FG_MinimumBias, uint32_t FG_threshold, uint32_t ZS_threshold,
                           int numberOfSamples,   int numberOfPresamples,
                           int numberOfSamplesHF, int numberOfPresamplesHF,
                           uint32_t minSignalThreshold=0, uint32_t PMT_NoiseThreshold=0,
                           bool upgrade_hb=false, bool upgrade_he=false, bool upgrade_hf=false);
  ~HcalTriggerPrimitiveAlgo();

  template<typename... Digis>
  void run(const HcalTPGCoder* incoder,
           const HcalTPGCompressor* outcoder,
           HcalTrigPrimDigiCollection& result,
           const HcalTrigTowerGeometry* trigTowerGeometry,
           float rctlsb, const HcalFeatureBit* LongvrsShortCut,
           const Digis&... digis);

  template<typename T, typename... Args>
  void addDigis(const T& collection, const Args&... digis) {
     addDigis(collection);
     addDigis(digis...);
  };

  template<typename T>
  void addDigis(const T& collection) {
     for (const auto& digi: collection) {
        addSignal(digi);
     }
  };

  template<typename D>
  void addDigis(const HcalDataFrameContainer<D>& collection) {
     for (auto i = collection.begin(); i != collection.end(); ++i) {
        D digi(*i);
        addSignal(digi);
     }
  };

  void runZS(HcalTrigPrimDigiCollection& tp);
  void runFEFormatError(const FEDRawDataCollection* rawraw,
                        const HcalElectronicsMap* emap,
                        HcalTrigPrimDigiCollection & result);
  void setPeakFinderAlgorithm(int algo);
  void setNCTScaleShift(int);
  void setRCTScaleShift(int);

 private:

  /// adds the signal to the map
  void addSignal(const HBHEDataFrame & frame);
  void addSignal(const HFDataFrame & frame);
  void addSignal(const QIE10DataFrame& frame);
  void addSignal(const QIE11DataFrame& frame);
  void addSignal(const IntegerCaloSamples & samples);
  void addFG(const HcalTrigTowerDetId& id, std::vector<bool>& msb);

  /// adds the actual RecHits
  void analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  // Phase1: QIE11
  void analyzePhase1(IntegerCaloSamples& samples, HcalTriggerPrimitiveDigi& result);
  // Version 0: RCT
  void analyzeHF(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result, const int hf_lumi_shift);
  // Version 1: 1x1
  void analyzeHF2016(
          const IntegerCaloSamples& SAMPLES,
          HcalTriggerPrimitiveDigi& result,
          const int HF_LUMI_SHIFT,
          const HcalFeatureBit* HCALFEM
          );
  // With dual anode readout
  void analyzeHFPhase1(
          const IntegerCaloSamples& SAMPLES,
          HcalTriggerPrimitiveDigi& result,
          const int HF_LUMI_SHIFT,
          const HcalFeatureBit* HCALFEM
          );

   // Member initialized by constructor
  const HcaluLUTTPGCoder* incoder_;
  const HcalTPGCompressor* outcoder_;
  double theThreshold;
  bool peakfind_;
  std::vector<double> weights_;
  int latency_;
  bool FG_MinimumBias_;
  uint32_t FG_threshold_;
  uint32_t ZS_threshold_;
  int ZS_threshold_I_;
  int numberOfSamples_;
  int numberOfPresamples_;
  int numberOfSamplesHF_;
  int numberOfPresamplesHF_;
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

  const HcalTrigTowerGeometry * theTrigTowerGeometry;

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;  

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
  };
  typedef std::map<HcalTrigTowerDetId, std::map<uint32_t, std::array<HFUpgradeDetails, 4>>> HFUpgradeDetailMap;
  HFUpgradeDetailMap theHFUpgradeDetailMap;
  
  typedef std::vector<IntegerCaloSamples> SumFGContainer;
  typedef std::map< HcalTrigTowerDetId, SumFGContainer > TowerMapFGSum;
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
  typedef std::map<uint32_t, std::vector<bool> > TowerMapVeto;
  TowerMapVeto HF_Veto;

  typedef std::map<HcalTrigTowerDetId, std::vector<bool> > FGbitMap;
  FGbitMap fgMap_;

  bool upgrade_hb_;
  bool upgrade_he_;
  bool upgrade_hf_;

  static const int first_he_tower = 16;
};

template<typename... Digis>
void HcalTriggerPrimitiveAlgo::run(const HcalTPGCoder* incoder,
                                   const HcalTPGCompressor* outcoder,
                                   HcalTrigPrimDigiCollection& result,
                                   const HcalTrigTowerGeometry* trigTowerGeometry,
                                   float rctlsb, const HcalFeatureBit* LongvrsShortCut,
                                   const Digis&... digis) {
   theTrigTowerGeometry = trigTowerGeometry;
    
   incoder_=dynamic_cast<const HcaluLUTTPGCoder*>(incoder);
   outcoder_=outcoder;

   theSumMap.clear();
   theTowerMapFGSum.clear();
   HF_Veto.clear();
   fgMap_.clear();
   theHFDetailMap.clear();
   theHFUpgradeDetailMap.clear();

   // Add all digi collections
   addDigis(digis...);

   // VME produces additional bits on the front used by lumi but not the
   // trigger, this shift corrects those out by right shifting over them.
   for(SumMap::iterator mapItr = theSumMap.begin(); mapItr != theSumMap.end(); ++mapItr) {
      result.push_back(HcalTriggerPrimitiveDigi(mapItr->first));
      HcalTrigTowerDetId detId(mapItr->second.id());
      if(detId.ietaAbs() >= theTrigTowerGeometry->firstHFTower(detId.version())) { 
         if (detId.version() == 0) {
            analyzeHF(mapItr->second, result.back(), RCTScaleShift);
         } else if (detId.version() == 1) {
            if (upgrade_hf_)
               analyzeHFPhase1(mapItr->second, result.back(), NCTScaleShift, LongvrsShortCut);
            else
               analyzeHF2016(mapItr->second, result.back(), NCTScaleShift, LongvrsShortCut);
         } else {
            // Things are going to go poorly
         }
      }
      else {
         if (upgrade_he_ and abs(detId.ieta()) >= first_he_tower) {
            analyzePhase1(mapItr->second, result.back());
         } else if (upgrade_hb_ and detId.subdet() < first_he_tower) {
            analyzePhase1(mapItr->second, result.back());
         } else {
            analyze(mapItr->second, result.back());
         }
      }
   }

   // Free up some memory
   theSumMap.clear();
   theTowerMapFGSum.clear();
   HF_Veto.clear();
   fgMap_.clear();
   theHFDetailMap.clear();
   theHFUpgradeDetailMap.clear();

   return;
}

#endif
