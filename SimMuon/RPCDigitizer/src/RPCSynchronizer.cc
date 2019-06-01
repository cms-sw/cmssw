#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "CLHEP/Random/RandGaussQ.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

RPCSynchronizer::RPCSynchronizer(const edm::ParameterSet& config) {
  resRPC = config.getParameter<double>("timeResolution");
  timOff = config.getParameter<double>("timingRPCOffset");
  dtimCs = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle = config.getParameter<double>("timeJitter");
  sspeed = config.getParameter<double>("signalPropagationSpeed");
  lbGate = config.getParameter<double>("linkGateWidth");
  LHCGate = config.getParameter<double>("Gate");
  cosmics = config.getParameter<bool>("cosmics");
  irpc_timing_res = config.getParameter<double>("IRPC_time_resolution");
  irpc_electronics_jitter = config.getParameter<double>("IRPC_electronics_jitter");
  N_BX = config.getParameter<int>("BX_range");
  //"magic" parameter for cosmics
  cosmicPar = 37.62;

  double c = 299792458;  // [m/s]
  //light speed in [cm/ns]
  cspeed = c * 1e+2 * 1e-9;
  //signal propagation speed [cm/ns]
  sspeed = sspeed * cspeed;
}

RPCSynchronizer::~RPCSynchronizer() {}

int RPCSynchronizer::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine) {
  RPCSimSetUp* simsetup = this->getRPCSimSetUp();
  const RPCGeometry* geometry = simsetup->getGeometry();
  float timeref = simsetup->getTime(simhit->detUnitId());

  int bx = -999;
  LocalPoint simHitPos = simhit->localPosition();
  float tof = simhit->timeOfFlight();

  //automatic variable to prevent memory leak

  float rr_el = CLHEP::RandGaussQ::shoot(engine, 0., resEle);

  RPCDetId SimDetId(simhit->detUnitId());

  const RPCRoll* SimRoll = nullptr;

  for (TrackingGeometry::DetContainer::const_iterator it = geometry->dets().begin(); it != geometry->dets().end();
       it++) {
    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      auto ch = dynamic_cast<const RPCChamber*>(*it);

      std::vector<const RPCRoll*> rollsRaf = (ch->rolls());
      for (std::vector<const RPCRoll*>::iterator r = rollsRaf.begin(); r != rollsRaf.end(); ++r) {
        if ((*r)->id() == SimDetId) {
          SimRoll = &(*(*r));
          break;
        }
      }
    }
  }

  if (SimRoll != nullptr) {
    float distanceFromEdge = 0;
    float half_stripL = 0.;

    if (SimRoll->id().region() == 0) {
      const RectangularStripTopology* top_ = dynamic_cast<const RectangularStripTopology*>(&(SimRoll->topology()));
      half_stripL = top_->stripLength() / 2;
      distanceFromEdge = half_stripL + simHitPos.y();
    } else {
      const TrapezoidalStripTopology* top_ = dynamic_cast<const TrapezoidalStripTopology*>(&(SimRoll->topology()));
      half_stripL = top_->stripLength() / 2;
      distanceFromEdge = half_stripL - simHitPos.y();
    }

    float prop_time = distanceFromEdge / sspeed;

    double rr_tim1 = CLHEP::RandGaussQ::shoot(engine, 0., resRPC);
    double total_time = tof + prop_time + timOff + rr_tim1 + rr_el;

    // Bunch crossing assignment
    double time_differ = 0.;

    if (cosmics) {
      time_differ = (total_time - (timeref + ((half_stripL / sspeed) + timOff))) / cosmicPar;
    } else if (!cosmics) {
      time_differ = total_time - (timeref + (half_stripL / sspeed) + timOff);
    }

    double inf_time = 0;
    double sup_time = 0;

    for (int n = -N_BX; n <= N_BX; ++n) {
      if (cosmics) {
        inf_time = (-lbGate / 2 + n * LHCGate) / cosmicPar;
        sup_time = (lbGate / 2 + n * LHCGate) / cosmicPar;
      } else if (!cosmics) {
        inf_time = -lbGate / 2 + n * LHCGate;
        sup_time = lbGate / 2 + n * LHCGate;
      }

      if (inf_time < time_differ && time_differ < sup_time) {
        bx = n;
        break;
      }
    }
  }

  return bx;
}

int RPCSynchronizer::getSimHitBxAndTimingForIRPC(const PSimHit* simhit, CLHEP::HepRandomEngine* engine) {
  RPCSimSetUp* simsetup = this->getRPCSimSetUp();
  const RPCGeometry* geometry = simsetup->getGeometry();
  float timeref = simsetup->getTime(simhit->detUnitId());

  int bx = -999;
  LocalPoint simHitPos = simhit->localPosition();
  float tof = simhit->timeOfFlight();

  //automatic variable to prevent memory leak

  //  float rr_el = CLHEP::RandGaussQ::shoot(engine, 0.,resEle);
  float rr_el = CLHEP::RandGaussQ::shoot(engine, 0., irpc_electronics_jitter);

  RPCDetId SimDetId(simhit->detUnitId());

  const RPCRoll* SimRoll = nullptr;

  for (TrackingGeometry::DetContainer::const_iterator it = geometry->dets().begin(); it != geometry->dets().end();
       it++) {
    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      auto ch = dynamic_cast<const RPCChamber*>(*it);

      std::vector<const RPCRoll*> rollsRaf = (ch->rolls());
      for (std::vector<const RPCRoll*>::iterator r = rollsRaf.begin(); r != rollsRaf.end(); ++r) {
        if ((*r)->id() == SimDetId) {
          SimRoll = &(*(*r));
          break;
        }
      }
    }
  }

  if (SimRoll != nullptr) {
    float distanceFromEdge = 0;
    float half_stripL = 0.;

    if (SimRoll->id().region() == 0) {
      const RectangularStripTopology* top_ = dynamic_cast<const RectangularStripTopology*>(&(SimRoll->topology()));
      half_stripL = top_->stripLength() / 2;
      distanceFromEdge = half_stripL + simHitPos.y();
    } else {
      const TrapezoidalStripTopology* top_ = dynamic_cast<const TrapezoidalStripTopology*>(&(SimRoll->topology()));
      half_stripL = top_->stripLength() / 2;
      distanceFromEdge = half_stripL - simHitPos.y();
    }

    float prop_time = distanceFromEdge / sspeed;

    //    double rr_tim1 = CLHEP::RandGaussQ::shoot(engine, 0.,resRPC);
    double rr_tim1 = CLHEP::RandGaussQ::shoot(engine, 0., irpc_timing_res);

    double total_time = tof + prop_time + timOff + rr_tim1 + rr_el;

    // Bunch crossing assignment
    double time_differ = 0.;

    if (cosmics) {
      time_differ = (total_time - (timeref + ((half_stripL / sspeed) + timOff))) / cosmicPar;
    } else if (!cosmics) {
      time_differ = total_time - (timeref + (half_stripL / sspeed) + timOff);
    }

    double exact_total_time = tof + prop_time + timOff;
    double exact_time_differ = 0.;

    if (cosmics) {
      exact_time_differ = (exact_total_time - (timeref + ((half_stripL / sspeed) + timOff))) / cosmicPar;
    } else if (!cosmics) {
      exact_time_differ = exact_total_time - (timeref + (half_stripL / sspeed) + timOff);
    }

    double inf_time = 0;
    double sup_time = 0;

    for (int n = -N_BX; n <= N_BX; ++n) {
      if (cosmics) {
        inf_time = (-lbGate / 2 + n * LHCGate) / cosmicPar;
        sup_time = (lbGate / 2 + n * LHCGate) / cosmicPar;
      } else if (!cosmics) {
        inf_time = -lbGate / 2 + n * LHCGate;
        sup_time = lbGate / 2 + n * LHCGate;
      }

      if (inf_time < time_differ && time_differ < sup_time) {
        bx = n;
        the_exact_time = exact_time_differ;
        the_smeared_time = time_differ;

        //cout<<"Debug\t"<<inf_time<<'\t'<<sup_time<<endl;
        ////if(bx)
        //	cout<<"Bingo\t"<<time_differ<<'\t'<<bx<<'\t'<<exact_time_differ<<'\t'<<exact_time_differ-time_differ<<'\t'<<exact_time_differ-bx*25.<<endl;
        break;
      }
    }
  }
  return bx;
}
