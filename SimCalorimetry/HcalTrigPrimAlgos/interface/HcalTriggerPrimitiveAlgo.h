#ifndef HcalSimAlgos_HcalTriggerPrimitiveAlgo_h
#define HcalSimAlgos_HcalTriggerPrimitiveAlgo_h

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"

#include <map>
#include <vector>
class CaloGeometry;
class IntegerCaloSamples;

class HcalTriggerPrimitiveAlgo {
public:
  HcalTriggerPrimitiveAlgo(bool pf, 
			   const std::vector<double>& w, int latency, uint32_t FG_threshold, uint32_t ZS_threshold, int firstTPSample, int TPSize);
  ~HcalTriggerPrimitiveAlgo();

  void run(const HcalTPGCoder * incoder,
	   const HcalTPGCompressor * outcoder,
	   const HBHEDigiCollection & hbheDigis,
           const HFDigiCollection & hfDigis,
           HcalTrigPrimDigiCollection & result);
 private:

  /// adds the signal to the map
  void addSignal(const HBHEDataFrame & frame);
  void addSignal(const HFDataFrame & frame);
  void addSignal(const IntegerCaloSamples & samples);
  void addSignalFG(const IntegerCaloSamples & samples);
  /// adds the actual RecHits
  void analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  void analyzeHF(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  void runZS(HcalTriggerPrimitiveDigi & tp);
 
  std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & id) const;

  HcalTrigTowerGeometry theTrigTowerGeometry; // from event setup eventually?

  const HcalTPGCoder * incoder_;
  const HcalTPGCompressor * outcoder_;

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;  
  
  typedef std::map<uint32_t, IntegerCaloSamples> SumMapFG;
  SumMapFG theFGSumMap;

  typedef std::multimap<HcalTrigTowerDetId, IntegerCaloSamples> TowerMapFG;
  TowerMapFG theTowerMapFG;

  double theThreshold;
  bool peakfind_;
  std::vector<double> weights_;
  int latency_;
  uint32_t FG_threshold_;
  uint32_t ZS_threshold_;
  int firstTPSample_;
  int TPSize_;
};


#endif



























