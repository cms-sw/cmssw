#ifndef HcalSimAlgos_HcalTriggerPrimitiveAlgo_h
#define HcalSimAlgos_HcalTriggerPrimitiveAlgo_h

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGTranscoder.h"

#include <map>
#include <vector>
class CaloGeometry;
class IntegerCaloSamples;

class HcalCoderFactory;

class HcalTriggerPrimitiveAlgo {
public:
  HcalTriggerPrimitiveAlgo(const HcalCoderFactory * coderFactory);
  ~HcalTriggerPrimitiveAlgo();

  void run(const HBHEDigiCollection & hbheDigis,
           const HFDigiCollection & hfDigis,
           HcalTrigPrimDigiCollection & result);

private:

  /// adds the signal to the map
  void addSignal(const HBHEDataFrame & frame);
  void addSignal(const HFDataFrame & frame);
  void addSignal(const IntegerCaloSamples & samples);

  /// adds the actual RecHits
  void analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  void outputMaker(const IntegerCaloSamples & samples, 
		   HcalTriggerPrimitiveDigi & result, 
		   const std::vector<bool> & finegrain);

  std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & id) const;

  HcalTrigTowerGeometry theTrigTowerGeometry; // from event setup eventually?

  const HcalTPGCoder * incoder_;
  HcalTPGTranscoder * outcoder_;

  const HcalCoderFactory * theCoderFactory;

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;  

  double theThreshold;
};


#endif



























