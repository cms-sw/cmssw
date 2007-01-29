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
			   const std::vector<double>& w, int latency);
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

  /// adds the actual RecHits
  void analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  void analyzeHF(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
 
  std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & id) const;

  HcalTrigTowerGeometry theTrigTowerGeometry; // from event setup eventually?

  const HcalTPGCoder * incoder_;
  const HcalTPGCompressor * outcoder_;

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;  

  double theThreshold;
  bool peakfind_;
  std::vector<double> weights_;
  int latency_;
};


#endif



























