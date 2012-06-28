#ifndef HcalUpgradeTriggerPrimitiveAlgo_h
#define HcalUpgradeTriggerPrimitiveAlgo_h

#include <map>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"

class CaloGeometry;
class IntegerCaloSamples;

class HcalUpgradeTriggerPrimitiveAlgo {
public:
  
  //------------------------------------------------------
  // Constructor/destructor
  //------------------------------------------------------
  
  HcalUpgradeTriggerPrimitiveAlgo(bool pf, const std::vector<double>& w, int latency, 
			       uint32_t FBThreshold, uint32_t ZSThreshold, uint32_t MinSignalThreshold,  uint32_t PMT_NoiseThreshold,
			       int numberOfSamples, int numberOfPresamples, bool excludeDepth5 );
  
  ~HcalUpgradeTriggerPrimitiveAlgo();
 
  //------------------------------------------------------
  // Event-by-event algorithm run function
  //------------------------------------------------------
 
  void run(const HcalTPGCoder * incoder,
	   const HcalTPGCompressor * outcoder,
	   const HBHEDigiCollection & hbheDigis,
           const HFDigiCollection & hfDigis,
           HcalUpgradeTrigPrimDigiCollection & result);
 private:
  
  //------------------------------------------------------
  // Mapping typedefs
  //------------------------------------------------------
  
  typedef std::map<HcalTrigTowerDetId, IntegerCaloSamples> SumMap;

  //------------------------------------------------------
  /// These convert from HBHEDataFrame & HFDataFrame to 
  //  linear IntegerCaloSamples and add the linear signal to the map
  //------------------------------------------------------
  
  void addSignal  (const HBHEDataFrame      & frame  );
  void addSignal  (const HFDataFrame        & frame  );
  void addSignal  (const IntegerCaloSamples & samples);

  //------------------------------------------------------
  /// These convert from linear IntegerCaloSamples to
  // compressed trigger primitive digis
  //------------------------------------------------------
  
  void analyze  (IntegerCaloSamples & samples, HcalUpgradeTriggerPrimitiveDigi & result);
  void analyzeHF(IntegerCaloSamples & samples, HcalUpgradeTriggerPrimitiveDigi & result);  
  
  //------------------------------------------------------
  // The LUT tables don't work for the SLHC software yet.
  // Use this for now.
  //------------------------------------------------------
  
   void fillDepth1Frame ( const HBHEDataFrame & frame,
			 HBHEDataFrame & depth1_frame );

   void fillRightDepthSamples ( 
      const IntegerCaloSamples & depth1_sample,
      IntegerCaloSamples & sample );
   
   void adc2Linear(const HBHEDataFrame& frame, IntegerCaloSamples & sample );

  //------------------------------------------------------
  // Conversion methods for HB & HE (for analyze function)
  //------------------------------------------------------

  // Weighted sum
  bool doSampleSum      (const IntegerCaloSamples& inputSamples,
			 IntegerCaloSamples& summedSamples,
			 int outlength);
  
  // Collapse/peakfinding
  void doSampleCollapse (const IntegerCaloSamples& originalSamples,
			 const IntegerCaloSamples& summedSamples,
			 IntegerCaloSamples& collapsedSamples );
  
  // Compression using LUT's
  void doSampleCompress (const IntegerCaloSamples& etSamples,
			 const std::vector<int>& fineGrainSamples,
			 HcalUpgradeTriggerPrimitiveDigi & digi);

  //------------------------------------------------------
  // Trig tower geometry
  //------------------------------------------------------

  HcalTrigTowerGeometry theTrigTowerGeometry;

  //------------------------------------------------------
  // Coders
  //------------------------------------------------------

  const HcaluLUTTPGCoder * incoder_;
  const HcalTPGCompressor * outcoder_;

  //------------------------------------------------------
  // Energy sum mapping
  //------------------------------------------------------

  SumMap theSumMap;  
  
  //------------------------------------------------------
  // Python file input
  //------------------------------------------------------

  bool peakfind_;
  int peak_finder_algorithm_;
  std::vector<double> weights_;
  int latency_;

  // thresholds

  uint32_t thePFThreshold_;
  uint32_t theFGThreshold_;
  uint32_t theZSThreshold_;
  uint32_t theMinSignalThreshold_;
  uint32_t thePMTNoiseThreshold_;

  // Samples info
  
  int numberOfSamples_;
  int numberOfPresamples_;
  
  // Other
  
  bool excludeDepth5_;
};


#endif



























