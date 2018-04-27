#ifndef __SimFastTiming_FastTimingCommon_ETLDeviceSim_h__
#define __SimFastTiming_FastTimingCommon_ETLDeviceSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerTypes.h"

#include <tuple>

class ETLDeviceSim {
 public:
  ETLDeviceSim(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt) { }

  void getHitsResponse(const std::vector<std::tuple<int,uint32_t,float>  > &hitRefs, 
		       const edm::Handle<edm::PSimHitContainer> &hits,
		       const float bxTime, const float tofDelay,
		       ftl_digitizer::FTLSimHitDataAccumulator *simHitAccumulator);

  float getChargeForHit(const PSimHit& hit) const {
    return 1000.f*hit.energyLoss()*MIPPerMeV_;
  }


 private:
  float MIPPerMeV_;
};

#endif
