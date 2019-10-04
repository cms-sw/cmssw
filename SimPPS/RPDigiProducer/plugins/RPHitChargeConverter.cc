#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeDivider.h"
#include "SimPPS/RPDigiProducer/plugins/RPHitChargeConverter.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeCollectionDrifter.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearInduceChargeOnStrips.h"

RPHitChargeConverter::RPHitChargeConverter(const edm::ParameterSet &params, CLHEP::HepRandomEngine &eng, RPDetId det_id)
    : det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPVerbosity");
  theRPChargeDivider = std::make_unique<RPLinearChargeDivider>(params, eng, det_id);
  theRPChargeCollectionDrifter = std::make_unique<RPLinearChargeCollectionDrifter>(params, det_id);
  theRPInduceChargeOnStrips = std::make_unique<RPLinearInduceChargeOnStrips>(params, det_id);
}

RPHitChargeConverter::~RPHitChargeConverter() {}

simromanpot::strip_charge_map RPHitChargeConverter::processHit(const PSimHit &hit) {
  simromanpot::energy_path_distribution ions_along_path = theRPChargeDivider->divide(hit);
  if (verbosity_)
    edm::LogInfo("HitChargeConverter") << det_id_ << " clouds no generated on the path=" << ions_along_path.size()
                                       << "\n";
  return theRPInduceChargeOnStrips->Induce(theRPChargeCollectionDrifter->Drift(ions_along_path));
}
