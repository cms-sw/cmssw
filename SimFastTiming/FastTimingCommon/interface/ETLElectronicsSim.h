#ifndef __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__
#define __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerTypes.h"

namespace ftl = ftl_digitizer;

class ETLElectronicsSim {
 public:
  ETLElectronicsSim(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt) { }

  void runBTL(const ftl::FTLSimHitDataAccumulator& input,
	      BTLDigiCollection& output) const {}

  void runETL(const ftl::FTLSimHitDataAccumulator& input,
	      ETLDigiCollection& output) const;

  void runTrivialShaperBTL(BTLDataFrame &dataFrame, 
			   const ftl::FTLSimHitData& chargeColl,
			   const ftl::FTLSimHitData& toa) const {}

  void runTrivialShaperETL(ETLDataFrame &dataFrame, 
			   const ftl::FTLSimHitData& chargeColl,
			   const ftl::FTLSimHitData& toa) const;

  void updateOutputBTL(BTLDigiCollection &coll,
		       const ETLDataFrame& rawDataFrame) const {}

  void updateOutputETL(ETLDigiCollection &coll,
		       const ETLDataFrame& rawDataFrame) const;

 private:

  const bool debug_;

  // adc/tdc bitwidths
  const uint32_t adcNbits_, tdcNbits_; 

  // synthesized adc/tdc information
  const float adcSaturation_MIP_;
  const float adcLSB_MIP_;
  const float adcThreshold_MIP_;
  const float toaLSB_ns_;

};

#endif
