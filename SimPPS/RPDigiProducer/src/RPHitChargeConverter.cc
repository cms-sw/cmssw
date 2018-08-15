#include "SimPPS/RPDigiProducer/interface/RPHitChargeConverter.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearChargeDivider.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearChargeCollectionDrifter.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearInduceChargeOnStrips.h"


RPHitChargeConverter::RPHitChargeConverter(const edm::ParameterSet &params, CLHEP::HepRandomEngine& eng, RPDetId det_id)
  : params_(params), det_id_(det_id)
{
  verbosity_ = params.getParameter<int>("RPVerbosity");
  theRPChargeDivider = new RPLinearChargeDivider(params, eng, det_id);
  theRPChargeCollectionDrifter = new RPLinearChargeCollectionDrifter(params, det_id);
  theRPInduceChargeOnStrips = new RPLinearInduceChargeOnStrips(params, det_id);
}

RPHitChargeConverter::~RPHitChargeConverter()
{
  delete theRPChargeDivider;
  delete theRPChargeCollectionDrifter;
  delete theRPInduceChargeOnStrips;
}


SimRP::strip_charge_map RPHitChargeConverter::processHit(const PSimHit &hit)
{  
  SimRP::energy_path_distribution ions_along_path = theRPChargeDivider->divide(hit);
  if(verbosity_)
    std::cout<<"HitChargeConverter "<<det_id_<<" clouds no generated on the path="<<ions_along_path.size()<<std::endl;
  return theRPInduceChargeOnStrips->Induce(theRPChargeCollectionDrifter->Drift(ions_along_path));
}

