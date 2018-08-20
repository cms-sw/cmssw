#ifndef __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__
#define __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

namespace mtd = mtd_digitizer;

namespace CLHEP {
  class HepRandomEngine;
}

class ETLElectronicsSim {
 public:
  ETLElectronicsSim(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt) { }

  void run(const mtd::MTDSimHitDataAccumulator& input,
	   ETLDigiCollection& output,
	   CLHEP::HepRandomEngine *hre) const;

  void runTrivialShaper(ETLDataFrame &dataFrame, 
			const mtd::MTDSimHitData& chargeColl,
			const mtd::MTDSimHitData& toa) const;

  void updateOutput(ETLDigiCollection &coll,
		    const ETLDataFrame& rawDataFrame) const;

  static constexpr int dfSIZE = 5;

 private:

  const bool  debug_;
  const float bxTime_;
  const reco::FormulaEvaluator sigmaEta_;

  // adc/tdc bitwidths
  const uint32_t adcNbits_, tdcNbits_; 

  // synthesized adc/tdc information
  const float adcSaturation_MIP_;
  const float adcLSB_MIP_;
  const float adcThreshold_MIP_;
  const float toaLSB_ns_;

};

#endif
