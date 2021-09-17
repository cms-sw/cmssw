#include "SimPPS/PPSPixelDigiProducer/interface/RPixHitChargeConverter.h"

RPixHitChargeConverter::RPixHitChargeConverter(const edm::ParameterSet &params,
                                               CLHEP::HepRandomEngine &eng,
                                               uint32_t det_id,
                                               const PPSPixelTopology &ppt)
    : det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPixVerbosity");
  theRPixChargeDivider = std::make_unique<RPixLinearChargeDivider>(params, eng, det_id);
  theRPixChargeCollectionDrifter = std::make_unique<RPixLinearChargeCollectionDrifter>(params, det_id, ppt);
  theRPixChargeShare = std::make_unique<RPixChargeShare>(params, det_id, ppt);
}

std::map<unsigned short, double> RPixHitChargeConverter::processHit(const PSimHit &hit, const PPSPixelTopology &ppt) {
  std::vector<RPixEnergyDepositUnit> ions_along_path = theRPixChargeDivider->divide(hit);
  if (verbosity_)
    edm::LogInfo("PPS") << "RPixHitChargeConverter " << det_id_
                        << " clouds no generated on the path=" << ions_along_path.size();
  return theRPixChargeShare->Share(theRPixChargeCollectionDrifter->Drift(ions_along_path), ppt);
}
