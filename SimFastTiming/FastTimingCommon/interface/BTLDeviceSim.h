#ifndef __SimFastTiming_FastTimingCommon_BTLDeviceSim_h__
#define __SimFastTiming_FastTimingCommon_BTLDeviceSim_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include <tuple>

namespace CLHEP {
  class HepRandomEngine;
}

class BTLDeviceSim {
public:
  BTLDeviceSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

  void getEvent(const edm::Event& evt) {}

  void getEventSetup(const edm::EventSetup& evt);

  void getHitsResponse(const std::vector<std::tuple<int, uint32_t, float> >& hitRefs,
                       const edm::Handle<edm::PSimHitContainer>& hits,
                       mtd_digitizer::MTDSimHitDataAccumulator* simHitAccumulator,
                       CLHEP::HepRandomEngine* hre);

private:
  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
  const MTDGeometry* geom_;
  const MTDTopology* topo_;

  const float bxTime_;
  const float LightYield_;
  const float LightCollEff_;

  const float LightCollSlope_;
  const float PDE_;
  const float LCEpositionSlope_;
};

#endif
