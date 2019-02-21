#include "SimPPS/PPSPixelDigiProducer/interface/RPixHitChargeConverter.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeDivider.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeCollectionDrifter.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixChargeShare.h"


RPixHitChargeConverter::RPixHitChargeConverter(const edm::ParameterSet &params, CLHEP::HepRandomEngine& eng, uint32_t det_id)
  : params_(params), det_id_(det_id)
{
  verbosity_ = params.getParameter<int>("RPixVerbosity");
  theRPixChargeDivider = new RPixLinearChargeDivider(params, eng, det_id);
  theRPixChargeCollectionDrifter = new RPixLinearChargeCollectionDrifter(params, det_id);
  theRPixChargeShare = new RPixChargeShare(params, det_id);
}

RPixHitChargeConverter::~RPixHitChargeConverter()
{
  delete theRPixChargeDivider;
  delete theRPixChargeCollectionDrifter;
  delete theRPixChargeShare;
}

std::map<unsigned short, double, std::less<unsigned short> > RPixHitChargeConverter::processHit(const PSimHit &hit)
{  
  std::vector<RPixEnergyDepositUnit> ions_along_path = theRPixChargeDivider->divide(hit);
  if(verbosity_)
    edm::LogInfo("RPixHitChargeConverter")<<det_id_<<" clouds no generated on the path="<<ions_along_path.size();
  return theRPixChargeShare->Share(theRPixChargeCollectionDrifter->Drift(ions_along_path));
}

