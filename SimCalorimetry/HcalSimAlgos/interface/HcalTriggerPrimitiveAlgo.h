#ifndef HcalSimAlgos_HcalTriggerPrimitiveAlgo_h
#define HcalSimAlgos_HcalTriggerPrimitiveAlgo_h

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGSimpleTranscoder.h"

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
           HcalTriggerPrimitiveDigi & result);

private:

  /// adds the signal to the map
  void addSignal(const HBHEDataFrame & frame);
  void addSignal(const HFDataFrame & frame);
  void addSignal(const IntegerCaloSamples & samples);

  /// changes the signal to be in ET instead of E
  void transverseComponent(IntegerCaloSamples & samples, const  HcalTrigTowerDetId & id) const;

  /// adds the actual RecHits
  void analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result);
  void outputMaker(const IntegerCaloSamples & samples, 
		   HcalTriggerPrimitiveDigi & result, 
		   const std::vector<bool> & finegrain);

  std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & id) const;

  HcalTrigTowerGeometry theTrigTowerGeometry;
  const HcalTPGCoder * tcoder;
  HcalTPGSimpleTranscoder * transcoder;

  const HcalCoderFactory * theCoderFactory;
  // counts from 1
  double theSinThetaTable[33];

  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;
  SumMap theSumMap;  

  double theThreshold;
  double theHBHECalibrationConstant;
  double theHFCalibrationConstant;
};


#endif



























