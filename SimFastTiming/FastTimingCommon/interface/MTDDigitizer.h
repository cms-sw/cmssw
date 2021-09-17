#ifndef FastTimingSimProducers_FastTimingCommon_MTDDigitizer_h
#define FastTimingSimProducers_FastTimingCommon_MTDDigitizer_h

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerBase.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTraits.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Math/interface/liblogintpack.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <tuple>

namespace mtd_digitizer {

  namespace MTDHelpers {
    // index , det id, time
    typedef std::tuple<int, uint32_t, float> MTDCaloHitTuple_t;

    inline bool orderByDetIdThenTime(const MTDCaloHitTuple_t& a, const MTDCaloHitTuple_t& b) {
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
  }  // namespace MTDHelpers

  inline void saveSimHitAccumulator(PMTDSimAccumulator& simResult,
                                    const MTDSimHitDataAccumulator& simData,
                                    const float minCharge,
                                    const float maxCharge) {
    constexpr auto nEnergies = std::tuple_size<decltype(MTDCellInfo().hit_info)>::value;
    static_assert(nEnergies == PMTDSimAccumulator::Data::energyMask + 1,
                  "PMTDSimAccumulator bit pattern needs to be updated");
    static_assert(nSamples == PMTDSimAccumulator::Data::sampleMask,
                  "PMTDSimAccumulator bit pattern needs to be updated");

    const float minPackChargeLog = minCharge > 0.f ? std::log(minCharge) : -2;
    const float maxPackChargeLog = std::log(maxCharge);
    constexpr uint16_t base = PMTDSimAccumulator::Data::dataMask;

    simResult.reserve(simData.size());
    // mimicking the digitization
    for (const auto& elem : simData) {
      // store only non-zero
      for (size_t iEn = 0; iEn < nEnergies; ++iEn) {
        const auto& samples = elem.second.hit_info[iEn];
        for (size_t iSample = 0; iSample < nSamples; ++iSample) {
          if (samples[iSample] > minCharge) {
            unsigned short packed;
            if (iEn == 1 || iEn == 3) {
              // assuming linear range for tof of 0..25
              packed = samples[iSample] / PREMIX_MAX_TOF * base;
            } else {
              packed = logintpack::pack16log(samples[iSample], minPackChargeLog, maxPackChargeLog, base);
            }
            simResult.emplace_back(elem.first.detid_, elem.first.row_, elem.first.column_, iEn, iSample, packed);
          }
        }
      }
    }
  }

  inline void loadSimHitAccumulator(MTDSimHitDataAccumulator& simData,
                                    const PMTDSimAccumulator& simAccumulator,
                                    const float minCharge,
                                    const float maxCharge) {
    const float minPackChargeLog = minCharge > 0.f ? std::log(minCharge) : -2;
    const float maxPackChargeLog = std::log(maxCharge);
    constexpr uint16_t base = PMTDSimAccumulator::Data::dataMask;

    for (const auto& detIdIndexHitInfo : simAccumulator) {
      auto foo = simData.emplace(
          MTDCellId(detIdIndexHitInfo.detId(), detIdIndexHitInfo.row(), detIdIndexHitInfo.column()), MTDCellInfo());
      auto simIt = foo.first;
      auto& hit_info = simIt->second.hit_info;

      size_t iEn = detIdIndexHitInfo.energyIndex();
      size_t iSample = detIdIndexHitInfo.sampleIndex();

      if (iEn > PMTDSimAccumulator::Data::energyMask + 1 || iSample > PMTDSimAccumulator::Data::sampleMask)
        throw cms::Exception("MTDDigitixer::loadSimHitAccumulator")
            << "Index out of range: iEn = " << iEn << " iSample = " << iSample << std::endl;

      float value;
      if (iEn == 1 || iEn == 3) {
        value = static_cast<float>(detIdIndexHitInfo.data()) / base * PREMIX_MAX_TOF;
      } else {
        value = logintpack::unpack16log(detIdIndexHitInfo.data(), minPackChargeLog, maxPackChargeLog, base);
      }

      if (iEn == 0 || iEn == 2) {
        hit_info[iEn][iSample] += value;
      } else if (hit_info[iEn][iSample] == 0 || value < hit_info[iEn][iSample]) {
        // For iEn==1 the digitizers just set the TOF of the first SimHit
        hit_info[iEn][iSample] = value;
      }
    }
  }

  template <class Traits>
  class MTDDigitizer : public MTDDigitizerBase {
  public:
    typedef typename Traits::DeviceSim DeviceSim;
    typedef typename Traits::ElectronicsSim ElectronicsSim;
    typedef typename Traits::DigiCollection DigiCollection;

    MTDDigitizer(const edm::ParameterSet& config, edm::ProducesCollector producesCollector, edm::ConsumesCollector& iC)
        : MTDDigitizerBase(config, producesCollector, iC),
          geomToken_(iC.esConsumes()),
          geom_(nullptr),
          deviceSim_(config.getParameterSet("DeviceSimulation"), iC),
          electronicsSim_(config.getParameterSet("ElectronicsSimulation"), iC),
          maxSimHitsAccTime_(config.getParameter<uint32_t>("maxSimHitsAccTime")) {}

    ~MTDDigitizer() override {}

    /**
       @short handle SimHit accumulation
    */
    void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;
    void accumulate(edm::Handle<edm::PSimHitContainer> const& hits,
                    int bxCrossing,
                    CLHEP::HepRandomEngine* hre) override;
    // for premixing
    void accumulate(const PMTDSimAccumulator& simAccumulator) override;

    /**
       @short actions at the start/end of event
    */
    void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) override;

  private:
    void resetSimHitDataAccumulator() { MTDSimHitDataAccumulator().swap(simHitAccumulator_); }

    const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
    const MTDGeometry* geom_;

    // implementations
    DeviceSim deviceSim_;            // processes a given simhit into an entry in a MTDSimHitDataAccumulator
    ElectronicsSim electronicsSim_;  // processes a MTDSimHitDataAccumulator into a BTLDigiCollection/ETLDigiCollection

    //handle sim hits
    const int maxSimHitsAccTime_;
    MTDSimHitDataAccumulator simHitAccumulator_;
  };

  template <class Traits>
  void MTDDigitizer<Traits>::accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits, 0, hre);
  }

  template <class Traits>
  void MTDDigitizer<Traits>::accumulate(PileUpEventPrincipal const& e,
                                        edm::EventSetup const& c,
                                        CLHEP::HepRandomEngine* hre) {
    edm::Handle<edm::PSimHitContainer> simHits;
    e.getByLabel(inputSimHits_, simHits);
    accumulate(simHits, e.bunchCrossing(), hre);
  }

  template <class Traits>
  void MTDDigitizer<Traits>::accumulate(edm::Handle<edm::PSimHitContainer> const& hits,
                                        int bxCrossing,
                                        CLHEP::HepRandomEngine* hre) {
    using namespace MTDHelpers;

    //create list of tuples (pos in container, RECO DetId, time) to be sorted first
    int nchits = (int)hits->size();
    std::vector<MTDCaloHitTuple_t> hitRefs;
    hitRefs.reserve(nchits);
    for (int i = 0; i < nchits; ++i) {
      const auto& the_hit = hits->at(i);

      DetId id = the_hit.detUnitId();

      if (verbosity_ > 0) {
        edm::LogInfo("MTDDigitizer") << " i/p " << std::hex << the_hit.detUnitId() << std::dec << " o/p " << id.rawId()
                                     << std::endl;
      }

      if (0 != id.rawId()) {
        hitRefs.emplace_back(i, id.rawId(), the_hit.tof());
      }
    }
    std::sort(hitRefs.begin(), hitRefs.end(), MTDHelpers::orderByDetIdThenTime);

    deviceSim_.getHitsResponse(hitRefs, hits, &simHitAccumulator_, hre);

    hitRefs.clear();
  }

  template <class Traits>
  void MTDDigitizer<Traits>::accumulate(const PMTDSimAccumulator& simAccumulator) {
    loadSimHitAccumulator(simHitAccumulator_, simAccumulator, premixStage1MinCharge_, premixStage1MaxCharge_);
  }

  template <class Traits>
  void MTDDigitizer<Traits>::initializeEvent(edm::Event const& e, edm::EventSetup const& c) {
    geom_ = &c.getData(geomToken_);

    deviceSim_.getEvent(e);
    deviceSim_.getEventSetup(c);
    if (not premixStage1_) {
      electronicsSim_.getEvent(e);
      electronicsSim_.getEventSetup(c);
    }
  }

  template <class Traits>
  void MTDDigitizer<Traits>::finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) {
    if (premixStage1_) {
      auto simResult = std::make_unique<PMTDSimAccumulator>();
      saveSimHitAccumulator(*simResult, simHitAccumulator_, premixStage1MinCharge_, premixStage1MaxCharge_);
      e.put(std::move(simResult), digiCollection_);
    } else {
      auto digiCollection = std::make_unique<DigiCollection>();
      electronicsSim_.run(simHitAccumulator_, *digiCollection, hre);
      e.put(std::move(digiCollection), digiCollection_);
    }

    //release memory for next event
    resetSimHitDataAccumulator();
  }
}  // namespace mtd_digitizer

#endif
