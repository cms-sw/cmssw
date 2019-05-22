#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTiming.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include <cmath>

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

#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <utility>
#include <map>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPCSimModelTiming::RPCSimModelTiming(const edm::ParameterSet& config) : RPCSim(config) {
  aveEff = config.getParameter<double>("averageEfficiency");
  aveCls = config.getParameter<double>("averageClusterSize");
  resRPC = config.getParameter<double>("timeResolution");
  timOff = config.getParameter<double>("timingRPCOffset");
  dtimCs = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle = config.getParameter<double>("timeJitter");
  sspeed = config.getParameter<double>("signalPropagationSpeed");
  lbGate = config.getParameter<double>("linkGateWidth");
  rpcdigiprint = config.getParameter<bool>("printOutDigitizer");

  rate = config.getParameter<double>("Rate");
  nbxing = config.getParameter<int>("Nbxing");
  gate = config.getParameter<double>("Gate");
  frate = config.getParameter<double>("Frate");
  do_Y = config.getParameter<bool>("do_Y_coordinate");
  sigmaY = config.getParameter<double>("sigmaY");
  eledig = config.getParameter<bool>("digitizeElectrons");

  if (rpcdigiprint) {
    edm::LogInfo("RPC digitizer parameters") << "Average Efficiency        = " << aveEff;
    edm::LogInfo("RPC digitizer parameters") << "Average Cluster Size      = " << aveCls << " strips";
    edm::LogInfo("RPC digitizer parameters") << "RPC Time Resolution       = " << resRPC << " ns";
    edm::LogInfo("RPC digitizer parameters") << "RPC Signal formation time = " << timOff << " ns";
    edm::LogInfo("RPC digitizer parameters") << "RPC adjacent strip delay  = " << dtimCs << " ns";
    edm::LogInfo("RPC digitizer parameters") << "Electronic Jitter         = " << resEle << " ns";
    edm::LogInfo("RPC digitizer parameters") << "Signal propagation time   = " << sspeed << " x c";
    edm::LogInfo("RPC digitizer parameters") << "Link Board Gate Width     = " << lbGate << " ns";
  }

  _rpcSync = new RPCSynchronizer(config);
}

RPCSimModelTiming::~RPCSimModelTiming() { delete _rpcSync; }

void RPCSimModelTiming::simulate(const RPCRoll* roll,
                                 const edm::PSimHitContainer& rpcHits,
                                 CLHEP::HepRandomEngine* engine) {
  _rpcSync->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());

  RPCDetId rpcId = roll->id();
  RPCGeomServ RPCname(rpcId);

  const Topology& topology = roll->specs()->topology();

  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin(); _hit != rpcHits.end(); ++_hit) {
    if (!eledig && _hit->particleType() == 11)
      continue;
    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr = _hit->entryPoint();

    int time_hit = _rpcSync->getSimHitBxAndTimingForIRPC(&(*_hit), engine);
    double precise_time = _rpcSync->getSmearedTime();

    float posX = roll->strip(_hit->localPosition()) - static_cast<int>(roll->strip(_hit->localPosition()));

    std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());

    // Effinciecy
    int centralStrip = topology.channel(entr) + 1;
    ;
    float fire = CLHEP::RandFlat::shoot(engine);

    float smearedPositionY = CLHEP::RandGaussQ::shoot(engine, _hit->localPosition().y(), sigmaY);

    if (fire < veff[centralStrip - 1]) {
      int fstrip = centralStrip;
      int lstrip = centralStrip;

      // Compute the cluster size
      int clsize = this->getClSize(rpcId.rawId(), posX, engine);  // This is for cluster size chamber by chamber
      std::vector<int> cls;
      cls.push_back(centralStrip);
      if (clsize > 1) {
        for (int cl = 0; cl < (clsize - 1) / 2; cl++) {
          if (centralStrip - cl - 1 >= 1) {
            fstrip = centralStrip - cl - 1;
            cls.push_back(fstrip);
          }
          if (centralStrip + cl + 1 <= roll->nstrips()) {
            lstrip = centralStrip + cl + 1;
            cls.push_back(lstrip);
          }
        }
        if (clsize % 2 == 0) {
          // insert the last strip according to the
          // simhit position in the central strip
          int lr = LeftRightNeighbour(*roll, entr, centralStrip);
          if (lr == 1) {
            if (lstrip < roll->nstrips()) {
              lstrip++;
              cls.push_back(lstrip);
            }
          } else {
            if (fstrip > 1) {
              fstrip--;
              cls.push_back(fstrip);
            }
          }
        }
      }

      //digitize all the strips in the cluster
      //in the previuos version some strips were dropped
      //leading to un-physical "shift" of the cluster
      for (std::vector<int>::iterator i = cls.begin(); i != cls.end(); i++) {
        std::pair<int, int> digi(*i, time_hit);
        RPCDigi adigi(*i, time_hit);
        adigi.hasTime(true);
        adigi.setTime(precise_time);
        if (do_Y) {
          adigi.hasY(true);
          adigi.setY(smearedPositionY);
          adigi.setDeltaY(sigmaY);
        }
        irpc_digis.insert(adigi);
        theDetectorHitMap.insert(DetectorHitMap::value_type(digi, &(*_hit)));
      }
    }
  }
}

void RPCSimModelTiming::simulateNoise(const RPCRoll* roll, CLHEP::HepRandomEngine* engine) {
  RPCDetId rpcId = roll->id();
  RPCGeomServ RPCname(rpcId);
  std::vector<float> vnoise = (getRPCSimSetUp())->getNoise(rpcId.rawId());
  std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());
  unsigned int nstrips = roll->nstrips();
  double area = 0.0;
  float striplength, xmin, xmax;
  if (rpcId.region() == 0) {
    const RectangularStripTopology* top_ = dynamic_cast<const RectangularStripTopology*>(&(roll->topology()));
    xmin = (top_->localPosition(0.)).x();
    xmax = (top_->localPosition((float)roll->nstrips())).x();
    striplength = (top_->stripLength());
    area = striplength * (xmax - xmin);
  } else {
    const TrapezoidalStripTopology* top_ = dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
    xmin = (top_->localPosition(0.)).x();
    xmax = (top_->localPosition((float)roll->nstrips())).x();
    striplength = (top_->stripLength());
    area = striplength * (xmax - xmin);
  }

  for (unsigned int j = 0; j < vnoise.size(); ++j) {
    if (j >= nstrips)
      break;

    double ave = vnoise[j] * nbxing * gate * area * 1.0e-9 * frate / ((float)roll->nstrips());

    CLHEP::RandPoissonQ randPoissonQ(*engine, ave);
    N_hits = randPoissonQ.fire();
    for (int i = 0; i < N_hits; i++) {
      double precise_time = CLHEP::RandFlat::shoot(engine, (nbxing * gate) / gate);
      int time_hit = (static_cast<int>(precise_time)) - nbxing / 2;
      RPCDigi adigi(j + 1, time_hit);
      adigi.hasTime(true);
      adigi.setTime(precise_time);
      if (do_Y) {
        double positionY = CLHEP::RandFlat::shoot(engine, striplength);
        positionY -= striplength / 2;
        adigi.hasY(true);
        adigi.setY(positionY);
        adigi.setDeltaY(sigmaY);
      }
      irpc_digis.insert(adigi);
    }
  }
}

int RPCSimModelTiming::getClSize(uint32_t id, float posX, CLHEP::HepRandomEngine* engine) {
  std::vector<double> clsForDetId = getRPCSimSetUp()->getCls(id);

  int cnt = 1;
  int min = 1;
  double func = 0.0;
  std::vector<double> sum_clsize;

  sum_clsize.clear();
  sum_clsize = clsForDetId;
  int vectOffset(0);

  double rr_cl = CLHEP::RandFlat::shoot(engine);

  if (0.0 <= posX && posX < 0.2) {
    func = clsForDetId[19] * (rr_cl);
    vectOffset = 0;
  }
  if (0.2 <= posX && posX < 0.4) {
    func = clsForDetId[39] * (rr_cl);
    vectOffset = 20;
  }
  if (0.4 <= posX && posX < 0.6) {
    func = clsForDetId[59] * (rr_cl);
    vectOffset = 40;
  }
  if (0.6 <= posX && posX < 0.8) {
    func = clsForDetId[79] * (rr_cl);
    vectOffset = 60;
  }
  if (0.8 <= posX && posX < 1.0) {
    func = clsForDetId[89] * (rr_cl);
    vectOffset = 80;
  }

  for (int i = vectOffset; i < (vectOffset + 20); i++) {
    cnt++;
    if (func > clsForDetId[i]) {
      min = cnt;
    } else if (func < clsForDetId[i]) {
      break;
    }
  }
  return min;
}

int RPCSimModelTiming::LeftRightNeighbour(const RPCRoll& roll, const LocalPoint& hit_pos, int strip) {
  //if left return -1
  //if right return +1

  int leftStrip = strip - 1;
  int rightStrip = strip + 1;

  if (leftStrip < 0)
    return +1;
  if (rightStrip > roll.nstrips())
    return -1;

  double deltawL = fabs((roll.centreOfStrip(leftStrip)).x() - hit_pos.x());
  double deltawR = fabs((roll.centreOfStrip(rightStrip)).x() - hit_pos.x());

  if (deltawL >= deltawR) {
    return +1;
  } else {
    return -1;
  }
}
