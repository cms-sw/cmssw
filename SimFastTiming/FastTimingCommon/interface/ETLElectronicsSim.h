#ifndef __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__
#define __SimFastTiming_FastTimingCommon_ETLElectronicsSim_h__

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "SimFastTiming/FastTimingCommon/interface/ETLPulseShape.h"

namespace mtd = mtd_digitizer;

namespace CLHEP {
  class HepRandomEngine;
}

class ETLElectronicsSim {
public:
  ETLElectronicsSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

  void getEvent(const edm::Event& evt) {}

  void getEventSetup(const edm::EventSetup& evt);

  void run(const mtd::MTDSimHitDataAccumulator& input, ETLDigiCollection& output, CLHEP::HepRandomEngine* hre) const;

  void runTrivialShaper(ETLDataFrame& dataFrame,
                        const mtd::MTDSimHitData& chargeColl,
                        const mtd::MTDSimHitData& toa1,
                        const mtd::MTDSimHitData& toa2,
                        const uint8_t row,
                        const uint8_t column) const;

  void updateOutput(ETLDigiCollection& coll, const ETLDataFrame& rawDataFrame) const;

  static constexpr int dfSIZE = 5;

private:
  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  const MTDGeometry* geom_;

  const float bxTime_;
  const float integratedLum_;

  const ETLPulseShape etlPulseShape_;

  // adc/tdc bitwidths
  const uint32_t adcNbits_, tdcNbits_;

  // synthesized adc/tdc information
  const float adcSaturation_MIP_;
  const float adcLSB_MIP_;
  const uint32_t adcBitSaturation_;
  const float adcThreshold_MIP_;
  const float iThreshold_MIP_;
  const float toaLSB_ns_;
  const uint32_t tdcBitSaturation_;
  const float referenceChargeColl_;
  const float noiseLevel_;
  const float sigmaDistorsion_;
  const float sigmaTDC_;
  const reco::FormulaEvaluator formulaLandauNoise_;
};

#endif
