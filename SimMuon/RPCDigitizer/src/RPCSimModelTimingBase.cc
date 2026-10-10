#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTimingBase.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <cmath>

RPCSimModelTimingBase::RPCSimModelTimingBase(const edm::ParameterSet& config)
    : RPCSim(config), rpcSync_(std::make_unique<RPCSynchronizer>(config)) {
  aveEff_ = config.getParameter<double>("averageEfficiency");
  aveCls_ = config.getParameter<double>("averageClusterSize");
  resRPC_ = config.getParameter<double>("timeResolution");
  timOff_ = config.getParameter<double>("timingRPCOffset");
  dtimCs_ = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle_ = config.getParameter<double>("timeJitter");
  sSpeed_ = config.getParameter<double>("signalPropagationSpeed");
  lbGate_ = config.getParameter<double>("linkGateWidth");
  rate_ = config.getParameter<double>("rate");
  nBXing_ = config.getParameter<int>("nBXing");
  gate_ = config.getParameter<double>("gate");
  fRate_ = config.getParameter<double>("fRate");

  if (config.existsAs<double>("sigmaY"))
    sigmaY_ = config.getParameter<double>("sigmaY");
  else
    sigmaY_ = 0.0;

  rpcDigiPrint_ = config.getParameter<bool>("printOutDigitizer");
  eleDig_ = config.getParameter<bool>("digitizeElectrons");

  if (rpcDigiPrint_) {
    edm::LogInfo("RPC digitizer parameters") << "Average Efficiency        = " << aveEff_;
    edm::LogInfo("RPC digitizer parameters") << "Average Cluster Size      = " << aveCls_;
    edm::LogInfo("RPC digitizer parameters") << "RPC Time Resolution       = " << resRPC_ << " ns";
    edm::LogInfo("RPC digitizer parameters") << "RPC Signal formation time = " << timOff_ << " ns";
    edm::LogInfo("RPC digitizer parameters") << "RPC adjacent strip delay  = " << dtimCs_ << " ns";
    edm::LogInfo("RPC digitizer parameters") << "Electronic Jitter         = " << resEle_ << " ns";
    edm::LogInfo("RPC digitizer parameters") << "Signal propagation speed  = " << sSpeed_;
    edm::LogInfo("RPC digitizer parameters") << "Link Board Gate Width     = " << lbGate_ << " ns";
  }
}

RPCSimModelTimingBase::~RPCSimModelTimingBase() = default;

//-------------------------------------------------------------
// Common preparation
//-------------------------------------------------------------
void RPCSimModelTimingBase::prepareSimulation(const RPCRoll* roll) {
  rpcSync_->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());
}

//-------------------------------------------------------------
// Main simulation algorithm
//-------------------------------------------------------------
void RPCSimModelTimingBase::simulate(const RPCRoll* roll,
                                     const edm::PSimHitContainer& rpcHits,
                                     CLHEP::HepRandomEngine* engine) {
  rpcSync_->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();

  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());

  const Topology& topology = roll->specs()->topology();

  //float striplength = getStripLength(roll);
  float striplength = stripLength(roll);
  for (const auto& hit : rpcHits) {
    if (!eleDig_ && hit.particleType() == 11)
      continue;

    simulateHit(roll,
                &hit,  // pointer to original PSimHit
                topology,
                striplength,
                engine);
  }
}

//-------------------------------------------------------------
// Simulation of one hit
//-------------------------------------------------------------
void RPCSimModelTimingBase::simulateHit(
    const RPCRoll* roll, const PSimHit* hit, const Topology& topology, float length, CLHEP::HepRandomEngine* engine) {
  RPCDetId rpcId = roll->id();
  const LocalPoint& entry = hit->entryPoint();

  int centralStrip = topology.channel(entry) + 1;
  float posX = roll->strip(hit->localPosition()) - static_cast<int>(roll->strip(hit->localPosition()));

  const std::vector<float>& efficiency = getRPCSimSetUp()->getEff(rpcId.rawId());

  float fire = CLHEP::RandFlat::shoot(engine);

  if (fire >= efficiency[centralStrip - 1])
    return;

  int clusterSize = getClSize(rpcId.rawId(), posX, engine);

  std::vector<int> cluster = buildCluster(roll, centralStrip, clusterSize, entry);

  float striplength = stripLength(roll);
  digitizeCluster(roll, *hit, cluster, striplength, engine);
}

//-------------------------------------------------------------
// Geometry
//-------------------------------------------------------------
float RPCSimModelTimingBase::stripLength(const RPCRoll* roll) const {
  if (auto rect = dynamic_cast<const RectangularStripTopology*>(&(roll->topology())))
    return rect->stripLength();

  if (auto trap = dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())))
    return trap->stripLength();

  return 0.0;
}

double RPCSimModelTimingBase::detectorArea(const RPCRoll* roll) const {
  float xmin = 0.;
  float xmax = 0.;
  float length = 0.;

  if (auto rect = dynamic_cast<const RectangularStripTopology*>(&(roll->topology()))) {
    xmin = rect->localPosition(0.).x();
    xmax = rect->localPosition((float)roll->nstrips()).x();
    length = rect->stripLength();
  } else if (auto trap = dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()))) {
    xmin = trap->localPosition(0.).x();
    xmax = trap->localPosition((float)roll->nstrips()).x();
    length = trap->stripLength();
  }
  return length * (xmax - xmin);
}

//-------------------------------------------------------------
// Cluster construction
//-------------------------------------------------------------
std::vector<int> RPCSimModelTimingBase::buildCluster(const RPCRoll* roll,
                                                     int centralStrip,
                                                     int clsize,
                                                     const LocalPoint& entry) {
  int firstStrip = centralStrip;
  int lastStrip = centralStrip;
  std::vector<int> cluster;
  cluster.push_back(centralStrip);

  if (clsize <= 1)
    return cluster;

  for (int cl = 0; cl < (clsize - 1) / 2; ++cl) {
    if (centralStrip - cl - 1 >= 1) {
      firstStrip = centralStrip - cl - 1;
      cluster.push_back(firstStrip);
    }

    if (centralStrip + cl + 1 <= roll->nstrips()) {
      lastStrip = centralStrip + cl + 1;
      cluster.push_back(lastStrip);
    }
  }

  if (clsize % 2 == 0) {
    int lr = LeftRightNeighbour(*roll, entry, centralStrip);
    if (lr == 1) {
      if (lastStrip < roll->nstrips())
        cluster.push_back(++lastStrip);
    } else {
      if (firstStrip > 1)
        cluster.push_back(--firstStrip);
    }
  }
  return cluster;
}

//-------------------------------------------------------------
// Cluster size
//-------------------------------------------------------------
int RPCSimModelTimingBase::getClSize(uint32_t id, float posX, CLHEP::HepRandomEngine* engine) {
  const std::vector<double>& cls = getRPCSimSetUp()->getCls(id);
  int offset = 0;
  double rnd = CLHEP::RandFlat::shoot(engine);
  double func = 0.;
  if (posX < 0.2) {
    func = cls[19] * rnd;
    offset = 0;
  } else if (posX < 0.4) {
    func = cls[39] * rnd;
    offset = 20;
  } else if (posX < 0.6) {
    func = cls[59] * rnd;
    offset = 40;
  } else if (posX < 0.8) {
    func = cls[79] * rnd;
    offset = 60;
  } else {
    func = cls[89] * rnd;
    offset = 80;
  }

  int result = 1;
  int cnt = 1;
  for (int i = offset; i < offset + 20; ++i) {
    ++cnt;
    if (func > cls[i])
      result = cnt;
    else
      break;
  }
  return result;
}

//-------------------------------------------------------------
// Left/right neighbour
//-------------------------------------------------------------
int RPCSimModelTimingBase::LeftRightNeighbour(const RPCRoll& roll, const LocalPoint& hit_pos, int strip) {
  int left = strip - 1;
  int right = strip + 1;

  if (left < 0)
    return +1;
  if (right > roll.nstrips())
    return -1;

  double dl = std::abs(roll.centreOfStrip(left).x() - hit_pos.x());
  double dr = std::abs(roll.centreOfStrip(right).x() - hit_pos.x());

  return (dl >= dr) ? +1 : -1;
}

//-------------------------------------------------------------
// Common noise simulation
//-------------------------------------------------------------
void RPCSimModelTimingBase::simulateNoise(const RPCRoll* roll, CLHEP::HepRandomEngine* engine) {
  RPCDetId rpcId = roll->id();
  const std::vector<float>& vnoise = getRPCSimSetUp()->getNoise(rpcId.rawId());
  unsigned int nstrips = roll->nstrips();
  double area = detectorArea(roll);

  for (unsigned int j = 0; j < vnoise.size(); ++j) {
    if (j >= nstrips)
      break;

    //-----------------------------------------------------
    // Average number of noise hits
    //-----------------------------------------------------
    double ave = vnoise[j] * nBXing_ * gate_ * area * 1.0e-9 * fRate_ / ((float)roll->nstrips());
    CLHEP::RandPoissonQ randPoissonQ(*engine, ave);
    int nHits = randPoissonQ.fire();

    //-----------------------------------------------------
    // Generate noise hits
    //-----------------------------------------------------
    createNoiseDigi(j + 1, nHits, engine);
  }
}
