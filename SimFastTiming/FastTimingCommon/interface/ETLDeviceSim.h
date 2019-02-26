#ifndef __SimFastTiming_FastTimingCommon_ETLDeviceSim_h__
#define __SimFastTiming_FastTimingCommon_ETLDeviceSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

#include <tuple>

namespace CLHEP {
  class HepRandomEngine;
}

class ETLDeviceSim {

 public:

  ETLDeviceSim(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt);

  void getHitsResponse(const std::vector<std::tuple<int,uint32_t,float> > &hitRefs, 
		       const edm::Handle<edm::PSimHitContainer> &hits,
		       mtd_digitizer::MTDSimHitDataAccumulator *simHitAccumulator,
		       CLHEP::HepRandomEngine *hre);

 private:

  const MTDGeometry* geom_;

  float MIPPerMeV_;
  float bxTime_;
  float tofDelay_;

};

#endif
