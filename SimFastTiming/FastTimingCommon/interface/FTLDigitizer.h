#ifndef FastTimingSimProducers_FastTimingCommon_FTLDigitizer_h
#define FastTimingSimProducers_FastTimingCommon_FTLDigitizer_h

#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerBase.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <tuple>

namespace ftl_digitizer {

  namespace FTLHelpers {
    // index , det id, time
    typedef std::tuple<int, uint32_t, float> FTLCaloHitTuple_t;

    bool orderByDetIdThenTime(const FTLCaloHitTuple_t& a, const FTLCaloHitTuple_t& b) {
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
  }  // namespace FTLHelpers

  template <class SensorPhysics, class ElectronicsSim>
  class FTLDigitizer : public FTLDigitizerBase {
  public:
    FTLDigitizer(const edm::ParameterSet& config, edm::ProducesCollector producesCollector, edm::ConsumesCollector& iC)
        : FTLDigitizerBase(config, producesCollector, iC),
          deviceSim_(config.getParameterSet("DeviceSimulation")),
          electronicsSim_(config.getParameterSet("ElectronicsSimulation")),
          maxSimHitsAccTime_(config.getParameter<uint32_t>("maxSimHitsAccTime")),
          bxTime_(config.getParameter<double>("bxTime")),
          tofDelay_(config.getParameter<double>("tofDelay")) {}

    ~FTLDigitizer() override {}

    /**
       @short handle SimHit accumulation
    */
    void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(edm::Handle<edm::PSimHitContainer> const& hits,
                    int bxCrossing,
                    CLHEP::HepRandomEngine* hre) override;

    /**
       @short actions at the start/end of event
    */
    void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;

    /**
       @short actions at the start/end of run
    */
    void beginRun(const edm::EventSetup& es) override;
    void endRun() override {}

  private:
    void resetSimHitDataAccumulator() { FTLSimHitDataAccumulator().swap(simHitAccumulator_); }

    // implementations
    SensorPhysics deviceSim_;        // processes a given simhit into an entry in a FTLSimHitDataAccumulator
    ElectronicsSim electronicsSim_;  // processes a FTLSimHitDataAccumulator into a FTLDigiCollection

    //handle sim hits
    const int maxSimHitsAccTime_;
    const double bxTime_;
    FTLSimHitDataAccumulator simHitAccumulator_;

    //delay to apply after evaluating time of arrival at the sensitive detector
    const float tofDelay_;

    //geometries
    std::unordered_set<DetId> validIds_;
    edm::ESWatcher<IdealGeometryRecord> idealGeomWatcher_;
    edm::ESHandle<FastTimeDDDConstants> dddFTL_;
  };

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::accumulate(edm::Event const& e,
                                                               edm::EventSetup const& c,
                                                               CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits, 0, hre);
  }

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::accumulate(PileUpEventPrincipal const& e,
                                                               edm::EventSetup const& c,
                                                               CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits, e.bunchCrossing(), hre);
  }

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::accumulate(edm::Handle<edm::PSimHitContainer> const& hits,
                                                               int bxCrossing,
                                                               CLHEP::HepRandomEngine* hre) {
    using namespace FTLHelpers;
    //configuration to apply for the computation of time-of-flight
    bool weightToAbyEnergy(false);
    float tdcOnset(0.f);

    //create list of tuples (pos in container, RECO DetId, time) to be sorted first
    int nchits = (int)hits->size();
    std::vector<FTLCaloHitTuple_t> hitRefs;
    hitRefs.reserve(nchits);
    for (int i = 0; i < nchits; ++i) {
      const auto& the_hit = hits->at(i);

      DetId id = (validIds_.count(the_hit.detUnitId()) ? the_hit.detUnitId() : 0);

      if (verbosity_ > 0) {
        edm::LogInfo("HGCDigitizer") << " i/p " << std::hex << the_hit.detUnitId() << std::dec << " o/p " << id.rawId()
                                     << std::endl;
      }

      if (0 != id.rawId()) {
        hitRefs.emplace_back(i, id.rawId(), the_hit.tof());
      }
    }
    std::sort(hitRefs.begin(), hitRefs.end(), FTLHelpers::orderByDetIdThenTime);

    //loop over sorted hits
    nchits = hitRefs.size();
    for (int i = 0; i < nchits; ++i) {
      const int hitidx = std::get<0>(hitRefs[i]);
      const uint32_t id = std::get<1>(hitRefs[i]);

      //get the data for this cell, if not available then we skip it

      if (!validIds_.count(id))
        continue;
      auto simHitIt = simHitAccumulator_.emplace(id, FTLCellInfo()).first;

      if (id == 0)
        continue;  // to be ignored at RECO level

      const float toa = std::get<2>(hitRefs[i]);
      const PSimHit& hit = hits->at(hitidx);
      const float charge = deviceSim_.getChargeForHit(hit);

      //distance to the center of the detector
      const float dist2center(0.1f * hit.entryPoint().mag());

      //hit time: [time()]=ns  [centerDist]=cm [refSpeed_]=cm/ns + delay by 1ns
      //accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
      const float tof = toa - dist2center / refSpeed_ + tofDelay_;
      const int itime = std::floor(tof / bxTime_) + 9;

      if (itime < 0 || itime > 14)
        continue;

      //check if time index is ok and store energy
      if (itime >= (int)simHitIt->second.hit_info[0].size())
        continue;

      (simHitIt->second).hit_info[0][itime] += charge;
      float accCharge = (simHitIt->second).hit_info[0][itime];

      //time-of-arrival (check how to be used)
      if (weightToAbyEnergy)
        (simHitIt->second).hit_info[1][itime] += charge * tof;
      else if ((simHitIt->second).hit_info[1][itime] == 0) {
        if (accCharge > tdcOnset) {
          //extrapolate linear using previous simhit if it concerns to the same DetId
          float fireTDC = tof;
          if (i > 0) {
            uint32_t prev_id = std::get<1>(hitRefs[i - 1]);
            if (prev_id == id) {
              float prev_toa = std::get<2>(hitRefs[i - 1]);
              float prev_tof(prev_toa - dist2center / refSpeed_ + tofDelay_);
              float deltaQ2TDCOnset = tdcOnset - ((simHitIt->second).hit_info[0][itime] - charge);
              float deltaQ = charge;
              float deltaT = (tof - prev_tof);
              fireTDC = deltaT * (deltaQ2TDCOnset / deltaQ) + prev_tof;
            }
          }

          (simHitIt->second).hit_info[1][itime] = fireTDC;
        }
      }
    }
    hitRefs.clear();
  }

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::initializeEvent(edm::Event const& e, edm::EventSetup const& c) {
    deviceSim_.getEvent(e);
    electronicsSim_.getEvent(e);
  }

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::finalizeEvent(edm::Event& e,
                                                                  edm::EventSetup const& c,
                                                                  CLHEP::HepRandomEngine* hre) {
    auto digiCollection = std::make_unique<FTLDigiCollection>();

    electronicsSim_.run(simHitAccumulator_, *digiCollection);

    e.put(std::move(digiCollection), digiCollection_);

    //release memory for next event
    resetSimHitDataAccumulator();
  }

  template <class SensorPhysics, class ElectronicsSim>
  void FTLDigitizer<SensorPhysics, ElectronicsSim>::beginRun(const edm::EventSetup& es) {
    if (idealGeomWatcher_.check(es)) {
      /// Get DDD constants
      es.get<IdealGeometryRecord>().get(dddFTL_);
      {  // force scope for the temporary nameless unordered_set
        std::unordered_set<DetId>().swap(validIds_);
      }
      // recalculate valid detids
      for (int zside = -1; zside <= 1; zside += 2) {
        for (unsigned type = 1; type <= 2; ++type) {
          for (unsigned izeta = 0; izeta < 1 << 10; ++izeta) {
            for (unsigned iphi = 0; iphi < 1 << 10; ++iphi) {
              if (dddFTL_->isValidXY(type, izeta, iphi)) {
                validIds_.emplace(FastTimeDetId(type, izeta, iphi, zside));
              }
            }
          }
        }
      }
      validIds_.reserve(validIds_.size());
    }
    deviceSim_.getEventSetup(es);
    electronicsSim_.getEventSetup(es);
  }
}  // namespace ftl_digitizer

#endif
