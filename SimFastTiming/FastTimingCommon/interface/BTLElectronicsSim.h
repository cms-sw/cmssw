#ifndef __SimFastTiming_FastTimingCommon_BTLElectronicsSim_h__
#define __SimFastTiming_FastTimingCommon_BTLElectronicsSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "SimFastTiming/FastTimingCommon/interface/BTLPulseShape.h"

namespace mtd = mtd_digitizer;

namespace CLHEP {
  class HepRandomEngine;
}

class BTLElectronicsSim {
 public:
  BTLElectronicsSim(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt) { }


  void run(const mtd::MTDSimHitDataAccumulator& input,
	   BTLDigiCollection& output,
	   CLHEP::HepRandomEngine *hre) const;

  void runTrivialShaper(BTLDataFrame &dataFrame, 
			const mtd::MTDSimHitData& chargeColl,
			const mtd::MTDSimHitData& toa1,
			const mtd::MTDSimHitData& toa2) const;

  void updateOutput(BTLDigiCollection &coll,
		    const BTLDataFrame& rawDataFrame) const;

  static constexpr int dfSIZE = 5;

 private:

  const bool debug_;

  //const bool  FluctuateNpe_;
  const float ScintillatorDecayTime_;
  const float ChannelTimeOffset_;
  const float smearChannelTimeOffset_;

  const float EnergyThreshold_;
  const float TimeThreshold1_;
  const float TimeThreshold2_;

  const float Npe_to_pC_;

  // adc/tdc bitwidths
  const uint32_t adcNbits_, tdcNbits_; 

  // synthesized adc/tdc information
  const float adcSaturation_MIP_;
  const float adcLSB_MIP_;
  const float adcThreshold_MIP_;
  const float toaLSB_ns_;

  const BTLPulseShape btlPulseShape_; 

};

#endif
