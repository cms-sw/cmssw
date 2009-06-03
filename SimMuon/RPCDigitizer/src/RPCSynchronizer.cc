#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/RandFlat.h>

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <cmath>

using namespace std;

RPCSynchronizer::RPCSynchronizer(const edm::ParameterSet& config){

  resRPC = config.getParameter<double>("timeResolution");
  timOff = config.getParameter<double>("timingRPCOffset");
  dtimCs = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle = config.getParameter<double>("timeJitter");
  sspeed = config.getParameter<double>("signalPropagationSpeed");
  lbGate = config.getParameter<double>("linkGateWidth");

  cosmics = config.getParameter<bool>("cosmics");

  //"magic" parameter for cosmics
  cosmicPar=37.62;

  double c=299792458;// [m/s]
  //light speed in [cm/ns]
  cspeed=c*1e+2*1e-9;
  //signal propagation speed [cm/ns]
  sspeed=sspeed*cspeed;


  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RPCDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  CLHEP::HepRandomEngine& rndEngine = rng->getEngine();
  flatDistribution_ = new CLHEP::RandFlat(rndEngine);

}


RPCSynchronizer::~RPCSynchronizer(){
  delete flatDistribution_;
  // if(gaussian_ != 0)  delete gaussian_; 
  // if(poissonDistribution_ != 0) delete poissonDistribution_;
}


int RPCSynchronizer::getSimHitBx(const PSimHit* simhit)
{

  RPCSimSetUp* simsetup = this->getRPCSimSetUp();
  const RPCGeometry * geometry = simsetup->getGeometry();
  float timeref = simsetup->getTime(simhit->detUnitId());

  int bx = -999;
  LocalPoint simHitPos = simhit->localPosition();
  float tof = simhit->timeOfFlight();

  //Defining a new engine local to this method for the two distributions defined below
  edm::Service<edm::RandomNumberGenerator> rnd;
  if ( ! rnd.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RPCDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rnd->getEngine();
  
  //automatic variable to prevent memory leak
  CLHEP::RandGaussQ gaussian1(engine,0.,resEle);
  
  float rr_el = gaussian1.fire();
  
  RPCDetId SimDetId(simhit->detUnitId());

  const RPCRoll* SimRoll = 0;

  for(TrackingGeometry::DetContainer::const_iterator it = geometry->dets().begin(); it != geometry->dets().end(); it++){
    
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      RPCDetId detId=ch->id();

      std::vector< const RPCRoll*> rollsRaf = (ch->rolls());
      for(std::vector<const RPCRoll*>::iterator r = rollsRaf.begin();
	  r != rollsRaf.end(); ++r){

	  if((*r)->id() == SimDetId) {
	    SimRoll = &(*(*r));
	    break;
	  }
      }
    }
  }

  if(SimRoll != 0){

    float distanceFromEdge = 0;
    float half_stripL = 0.;

    if(SimRoll->id().region() == 0){
      const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(SimRoll->topology()));
      half_stripL = top_->stripLength()/2;
      distanceFromEdge = half_stripL + simHitPos.y();
    }else{
      const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&(SimRoll->topology()));
      half_stripL = top_->stripLength()/2;
      distanceFromEdge = half_stripL - simHitPos.y();
    }


    float prop_time =  distanceFromEdge/sspeed;

    //automatic variable to prevent memory leak
    CLHEP::RandGaussQ gaussian2(engine,0.,resEle); 

    double rr_tim1 = gaussian2.fire();
    double total_time = tof + prop_time + timOff + rr_tim1 + rr_el;
    
    // Bunch crossing assignment
    double time_differ = 0.;

    if(cosmics){
      time_differ = (total_time - (timeref + ((half_stripL/sspeed ) + timOff)))/cosmicPar;
    }
    else if(!cosmics){
      time_differ = total_time - (timeref + ( half_stripL/sspeed ) + timOff);
    }
     
    double inf_time = 0;
    double sup_time = 0;


    for(int n = -5; n <= 5; ++n){

      if(cosmics){
	inf_time = (-lbGate/2 + n*lbGate )/cosmicPar;
	sup_time = ( lbGate/2 + n*lbGate )/cosmicPar;
      }
      else if(!cosmics){
	inf_time = -lbGate/2 + n*lbGate;
	sup_time =  lbGate/2 + n*lbGate;
      }

      if(inf_time < time_differ && time_differ < sup_time) {
	bx = n;
	break;
      }
    }
  }


  return bx;
}

