#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

ETLDeviceSim::ETLDeviceSim(const edm::ParameterSet& pset) : 
  MIPPerMeV_( 1.0/pset.getParameter<double>("meVPerMIP") ),
  bxTime_(pset.getParameter<double>("bxTime") ),
  tofDelay_(pset.getParameter<double>("tofDelay") ) {}


void ETLDeviceSim::getHitsResponse(const std::vector<std::tuple<int,uint32_t,float> > &hitRefs, 
				   const edm::Handle<edm::PSimHitContainer> &hits,
				   mtd_digitizer::MTDSimHitDataAccumulator *simHitAccumulator,
				   CLHEP::HepRandomEngine *hre){

  //loop over sorted hits
  const int nchits = hitRefs.size();
  for(int i=0; i<nchits; ++i) {
    const int hitidx   = std::get<0>(hitRefs[i]);
    const uint32_t id  = std::get<1>(hitRefs[i]);
    const MTDDetId detId(id);

    // Safety check
    if ( detId.det()!=DetId::Forward || detId.mtdSubDetector()!=2 ) continue;

    auto simHitIt = simHitAccumulator->emplace(id,mtd_digitizer::MTDCellInfo()).first;
      
    if(id==0) continue; // to be ignored at RECO level
      
    const float toa    = std::get<2>(hitRefs[i]) + tofDelay_;
    const PSimHit &hit = hits->at( hitidx );     
    const float charge = 1000.f*hit.energyLoss()*MIPPerMeV_;

    // Accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const int itime = std::floor( toa/bxTime_ ) + 9;
    if(itime<0 || itime>14) continue;     
      
    // Check if time index is ok and store energy
    if(itime >= (int)simHitIt->second.hit_info[0].size() ) continue;

    (simHitIt->second).hit_info[0][itime] += charge;
      
    // Store the time of the first SimHit in the right DataFrame bucket
    const float tof = toa - (itime-9)*bxTime_;

    if( (simHitIt->second).hit_info[1][itime] == 0. ||
	tof < (simHitIt->second).hit_info[1][itime] ) {
      (simHitIt->second).hit_info[1][itime] = tof;

    }

  }

}
