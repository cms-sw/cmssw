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
				   mtd_digitizer::MTDSimHitDataAccumulator *simHitAccumulator){

  bool weightToAbyEnergy(false);
  float tdcOnset(0.f);

  const float refSpeed = 0.1*CLHEP::c_light;

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
      
    const float toa    = std::get<2>(hitRefs[i]);
    const PSimHit &hit = hits->at( hitidx );     
    const float charge = 1000.f*hit.energyLoss()*MIPPerMeV_;

    //distance to the center of the detector
    const float dist2center( 0.1f*hit.entryPoint().mag() );
      
    //hit time: [time()]=ns  [centerDist]=cm [refSpeed]=cm/ns + delay by 1ns
    //accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const float tof = toa-dist2center/refSpeed+tofDelay_ ;
    const int itime = std::floor( tof/bxTime_ ) + 9;

    if(itime<0 || itime>14) continue;     
      
    //check if time index is ok and store energy
    if(itime >= (int)simHitIt->second.hit_info[0].size() ) continue;
      
    (simHitIt->second).hit_info[0][itime] += charge;
    float accCharge=(simHitIt->second).hit_info[0][itime];
      
    //time-of-arrival (check how to be used)
    if(weightToAbyEnergy) (simHitIt->second).hit_info[1][itime] += charge*tof;
    else if((simHitIt->second).hit_info[1][itime]==0)
      {	
	if( accCharge>tdcOnset )
	  {
	    //extrapolate linear using previous simhit if it concerns to the same DetId
	    float fireTDC=tof;
	    if(i>0)
	      {
		uint32_t prev_id = std::get<1>(hitRefs[i-1]);
		if(prev_id==id)
		  {
		    float prev_toa    = std::get<2>(hitRefs[i-1]);
		    float prev_tof(prev_toa-dist2center/refSpeed+tofDelay_);
		    float deltaQ2TDCOnset = tdcOnset-((simHitIt->second).hit_info[0][itime]-charge);
		    float deltaQ          = charge;
		    float deltaT          = (tof-prev_tof);
		    fireTDC               = deltaT*(deltaQ2TDCOnset/deltaQ)+prev_tof;
		  }		  
	      }
	      
	    (simHitIt->second).hit_info[1][itime]=fireTDC;

	  }
      }
  }


}
