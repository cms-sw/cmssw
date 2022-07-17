#ifndef HGCalSimProducers_HGCDigitizer_h
#define HGCalSimProducers_HGCDigitizer_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCalDigi/interface/PHGCSimAccumulator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <tuple>

class PCaloHit;
class PileUpEventPrincipal;

class HGCDigitizer {
public:
  HGCDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  ~HGCDigitizer() = default;

  // index , det id, time
  typedef std::tuple<int, uint32_t, float> HGCCaloHitTuple_t;
  static bool orderByDetIdThenTime(const HGCCaloHitTuple_t& a, const HGCCaloHitTuple_t& b) {
    unsigned int detId_a(std::get<1>(a)), detId_b(std::get<1>(b));

    if (detId_a < detId_b)
      return true;
    if (detId_a > detId_b)
      return false;

    double time_a(std::get<2>(a)), time_b(std::get<2>(b));
    if (time_a < time_b)
      return true;

    return false;
  }

  /**
     @short handle SimHit accumulation
   */
  void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);
  void accumulate_forPreMix(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);

  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);
  void accumulate_forPreMix(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);

  void accumulate(edm::Handle<edm::PCaloHitContainer> const& hits,
                  int bxCrossing,
                  const HGCalGeometry* geom,
                  CLHEP::HepRandomEngine* hre);
  void accumulate_forPreMix(edm::Handle<edm::PCaloHitContainer> const& hits,
                            int bxCrossing,
                            const HGCalGeometry* geom,
                            CLHEP::HepRandomEngine* hre);

  void accumulate_forPreMix(const PHGCSimAccumulator& simAccumulator, const bool minbiasFlag);
  /**
     @short actions at the start/end of event
   */
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);

  std::string digiCollection() { return digiCollection_; }

private:
  uint32_t getType() const;

  std::unique_ptr<hgc::HGCSimHitDataAccumulator> simHitAccumulator_;
  std::unique_ptr<hgc::HGCPUSimHitDataAccumulator> pusimHitAccumulator_;

  const std::string digiCollection_;

  //digitization type (it's up to the specializations to decide it's meaning)
  const int digitizationType_;

  // if true, we're running mixing in premixing stage1 and have to produce the output differently
  const bool premixStage1_;

  // Minimum charge threshold for premixing stage1
  const double premixStage1MinCharge_;
  // Maximum charge for packing in premixing stage1
  const double premixStage1MaxCharge_;

  //handle sim hits
  const int maxSimHitsAccTime_;
  const double bxTime_;
  double ev_per_eh_pair_;
  const std::string hitsProducer_;
  const std::string hitCollection_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> hitToken_;
  void resetSimHitDataAccumulator();
  void resetPUSimHitDataAccumulator();
  //debug position
  void checkPosition(const HGCalDigiCollection* digis) const;

  //digitizer
  std::unique_ptr<HGCDigitizerBase> theDigitizer_;

  //geometries
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESWatcher<CaloGeometryRecord> geomWatcher_;
  std::unordered_set<DetId> validIds_;
  const HGCalGeometry* gHGCal_ = nullptr;

  //misc switches
  const uint32_t verbosity_;

  //reference speed to evaluate time of arrival at the sensititive detector, assuming the center of CMS
  const float refSpeed_;

  //delay to apply after evaluating time of arrival at the sensitive detector
  const float tofDelay_;

  //average occupancies
  std::array<double, 4> averageOccupancies_;
  uint32_t nEvents_;

  //maxBx limit beyond which the Digitizer should filter out all hits
  static const unsigned int maxBx_ = 14;
  static const unsigned int thisBx_ = 9;
  std::vector<float> cce_;
  std::unordered_map<uint32_t, std::vector<std::pair<float, float>>> hitRefs_bx0;
  std::unordered_map<uint32_t, std::vector<std::tuple<float, float, float>>> PhitRefs_bx0;
  std::unordered_map<uint32_t, bool> hitOrder_monitor;
};

#endif
