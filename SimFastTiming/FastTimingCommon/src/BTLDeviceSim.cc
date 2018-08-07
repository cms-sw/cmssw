#include "SimFastTiming/FastTimingCommon/interface/BTLDeviceSim.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "CLHEP/Random/RandGaussQ.h"

BTLDeviceSim::BTLDeviceSim(const edm::ParameterSet& pset) : 
  bxTime_(pset.getParameter<double>("bxTime") ),
  LightYield_(pset.getParameter<double>("LightYield")),
  LightCollEff_(pset.getParameter<double>("LightCollectionEff")),
  LightCollTime_(pset.getParameter<double>("LightCollectionTime")),
  smearLightCollTime_(pset.getParameter<double>("smearLightCollectionTime")),
  PDE_(pset.getParameter<double>("PhotonDetectionEff")) { }


void BTLDeviceSim::getHitsResponse(const std::vector<std::tuple<int,uint32_t,float> > &hitRefs, 
				   const edm::Handle<edm::PSimHitContainer> &hits,
				   mtd_digitizer::MTDSimHitDataAccumulator *simHitAccumulator,
				   CLHEP::HepRandomEngine *hre){

  //loop over sorted simHits
  const int nchits = hitRefs.size();
  for(int ihit=0; ihit<nchits; ++ihit) {

    const int hitidx   = std::get<0>(hitRefs[ihit]);
    const uint32_t id  = std::get<1>(hitRefs[ihit]);
    const MTDDetId detId(id);
    const PSimHit &hit = hits->at( hitidx );     
    
    // --- Safety check on the detector ID
    if ( detId.det()!=DetId::Forward || detId.mtdSubDetector()!=1 ) continue;

    // --- Store the detector element ID as a key of the MTDSimHitDataAccumulator map
    auto simHitIt = simHitAccumulator->emplace(id,mtd_digitizer::MTDCellInfo()).first;

    // --- Get the simHit energy and convert it from MeV to photo-electrons
    float Npe = 1000.*hit.energyLoss()*LightYield_*LightCollEff_*PDE_;

    // --- Get the simHit time of arrival and add the light collection time
    float toa = std::get<2>(hitRefs[ihit]) + LightCollTime_;

    if ( smearLightCollTime_ > 0. )
      toa += CLHEP::RandGaussQ::shoot(hre, 0., smearLightCollTime_);

    // --- Accumulate in 15 buckets of 25 ns (9 pre-samples, 1 in-time, 5 post-samples)
    const int itime = std::floor( toa/bxTime_ ) + 9;
    if(itime<0 || itime>14) continue;     

    // --- Check if the time index is ok and accumulate the energy
    if(itime >= (int)simHitIt->second.hit_info[0].size() ) continue;

    (simHitIt->second).hit_info[0][itime] += Npe;

    // --- Store the time of the first SimHit in the right DataFrame bucket
    const float tof = toa - (itime-9)*bxTime_;

    if( (simHitIt->second).hit_info[1][itime] == 0 ||
	tof < (simHitIt->second).hit_info[1][itime] )
      (simHitIt->second).hit_info[1][itime] = tof;

  } // ihit loop

}
