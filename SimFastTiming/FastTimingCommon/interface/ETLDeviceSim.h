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

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include <tuple>

namespace CLHEP {
  class HepRandomEngine;
}

class ETLDeviceSim {
public:
  ETLDeviceSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

  void getEvent(const edm::Event& evt) {}

  void getEventSetup(const edm::EventSetup& evt);

  void getHitsResponse(const std::vector<std::tuple<int, uint32_t, float> >& hitRefs,
                       const edm::Handle<edm::PSimHitContainer>& hits,
                       mtd_digitizer::MTDSimHitDataAccumulator* simHitAccumulator,
                       CLHEP::HepRandomEngine* hre);

private:
  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  const MTDGeometry* geom_;
  float MIPPerMeV_;
  const float integratedLum_;
  const reco::FormulaEvaluator fluence_;
  const reco::FormulaEvaluator lgadGain_;
  const reco::FormulaEvaluator lgadGainDegradation_;
  const bool applyDegradation_;
  float bxTime_;
  float tofDelay_;
  const reco::FormulaEvaluator MPVMuon_;
  const reco::FormulaEvaluator MPVPion_;
  const reco::FormulaEvaluator MPVKaon_;
  const reco::FormulaEvaluator MPVElectron_;
  const reco::FormulaEvaluator MPVProton_;
};

#endif
