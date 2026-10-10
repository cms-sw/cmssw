#ifndef SimMuon_RPCDigitizer_RPCSimModelTimingBase_h
#define SimMuon_RPCDigitizer_RPCSimModelTimingBase_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"

#include <memory>
#include <vector>

class RPCRoll;
class Topology;

class RPCSimModelTimingBase : public RPCSim {

public:

  explicit RPCSimModelTimingBase(const edm::ParameterSet&);

  ~RPCSimModelTimingBase() override;

  //------------------------------------------------------
  // Common simulation algorithms
  //------------------------------------------------------
  void simulate(const RPCRoll*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) override;
  void simulateNoise(const RPCRoll*, CLHEP::HepRandomEngine*) override;

protected:
  void init() override {}

  //------------------------------------------------------
  // Detector-dependent digi creation
  //------------------------------------------------------
  virtual void digitizeCluster(const RPCRoll* roll, const PSimHit& hit, const std::vector<int>& cls, float striplength, CLHEP::HepRandomEngine* engine) = 0;
  virtual void createNoiseDigi(int strip, int hits, CLHEP::HepRandomEngine*) = 0;

  //------------------------------------------------------
  // Common hit processing
  //------------------------------------------------------
  void prepareSimulation(const RPCRoll*);
  void simulateHit(const RPCRoll*, const PSimHit*, const Topology&, float stripLength, CLHEP::HepRandomEngine*);

  //------------------------------------------------------
  // Geometry
  //------------------------------------------------------
  float stripLength(const RPCRoll*) const;
  double detectorArea(const RPCRoll*) const;


  //------------------------------------------------------
  // Cluster calculation
  //------------------------------------------------------
  std::vector<int> buildCluster(const RPCRoll*, int centralStrip, int clusterSize, const LocalPoint& hitPosition);
  int getClSize(uint32_t id, float posX, CLHEP::HepRandomEngine*);
  int LeftRightNeighbour(const RPCRoll&, const LocalPoint&, int strip);

protected:
  //------------------------------------------------------
  // Parameters
  //------------------------------------------------------
  double aveEff_;
  double aveCls_;
  double resRPC_;
  double timOff_;
  double dtimCs_;
  double resEle_;
  double sSpeed_;
  double lbGate_;
  double rate_;
  double gate_;
  double fRate_;
  double sigmaY_;
  int nBXing_;
  bool rpcDigiPrint_;
  bool eleDig_;

  //------------------------------------------------------
  // Shared objects
  //------------------------------------------------------
  std::unique_ptr<RPCSynchronizer> rpcSync_;
};

#endif

