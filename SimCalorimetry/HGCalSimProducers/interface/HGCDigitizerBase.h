#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <array>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

namespace hgc = hgc_digi;

namespace hgc_digi_utils {
  using hgc::HGCCellInfo;

  inline void addCellMetadata(HGCCellInfo& info, const HGCalGeometry* geom, const DetId& detid) {
    const auto& dddConst = geom->topology().dddConstants();
    bool isHalf = (((dddConst.geomMode() == HGCalGeometryMode::Hexagon) ||
                    (dddConst.geomMode() == HGCalGeometryMode::HexagonFull))
                       ? dddConst.isHalfCell(HGCalDetId(detid).wafer(), HGCalDetId(detid).cell())
                       : false);
    //base time samples for each DetId, initialized to 0
    info.size = (isHalf ? 0.5 : 1.0);
    info.thickness = 1 + dddConst.waferType(detid);
  }

  inline void addCellMetadata(HGCCellInfo& info, const CaloSubdetectorGeometry* geom, const DetId& detid) {
    if (DetId::Hcal == detid.det()) {
      const HcalGeometry* hc = static_cast<const HcalGeometry*>(geom);
      addCellMetadata(info, hc, detid);
    } else {
      const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
      addCellMetadata(info, hg, detid);
    }
  }

}  // namespace hgc_digi_utils

class HGCDigitizerBase {
public:
  typedef HGCalDataFrame DFr;
  typedef edm::SortedCollection<DFr> DColl;

  /**
     @short CTOR
  */
  HGCDigitizerBase(const edm::ParameterSet& ps);
  /**
     @short Gaussian Noise Generation Member Function
  */
  void GenerateGaussianNoise(CLHEP::HepRandomEngine* engine, const double NoiseMean, const double NoiseStd);
  /**
    @short steer digitization mode
 */
  void run(std::unique_ptr<DColl>& digiColl,
           hgc::HGCSimHitDataAccumulator& simData,
           const CaloSubdetectorGeometry* theGeom,
           const std::unordered_set<DetId>& validIds,
           uint32_t digitizationType,
           CLHEP::HepRandomEngine* engine);

  /**
     @short getters
  */
  float keV2fC() const { return keV2fC_; }
  bool toaModeByEnergy() const { return (myFEelectronics_->toaMode() == HGCFEElectronics<DFr>::WEIGHTEDBYE); }
  float tdcOnset() const { return myFEelectronics_->getTDCOnset(); }
  std::array<float, 3> tdcForToAOnset() const { return myFEelectronics_->getTDCForToAOnset(); }
  DetId::Detector det() const { return det_; }
  ForwardSubdetector subdet() const { return subdet_; }

  /**
     @short a trivial digitization: sum energies and digitize
   */
  void runSimple(std::unique_ptr<DColl>& coll,
                 hgc::HGCSimHitDataAccumulator& simData,
                 const CaloSubdetectorGeometry* theGeom,
                 const std::unordered_set<DetId>& validIds,
                 CLHEP::HepRandomEngine* engine);

  /**
     @short prepares the output according to the number of time samples to produce
  */
  void updateOutput(std::unique_ptr<DColl>& coll, const DFr& rawDataFrame);

  /**
     @short to be specialized by top class
  */
  virtual void runDigitizer(std::unique_ptr<DColl>& coll,
                            hgc::HGCSimHitDataAccumulator& simData,
                            const CaloSubdetectorGeometry* theGeom,
                            const std::unordered_set<DetId>& validIds,
                            CLHEP::HepRandomEngine* engine) = 0;
  /**
     @short DTOR
  */
  virtual ~HGCDigitizerBase(){};

protected:
  //baseline configuration
  edm::ParameterSet myCfg_;

  //1keV in fC
  float keV2fC_;

  //noise level (used if scaleByDose=False)
  std::vector<float> noise_fC_;

  //charge collection efficiency (used if scaleByDose=False)
  std::vector<double> cce_;

  //determines if the dose map should be used instead
  bool scaleByDose_;

  //multiplicative fator to scale fluence map
  double scaleByDoseFactor_;

  //path to dose map
  std::string doseMapFile_;

  //noise maps (used if scaleByDose=True)
  HGCalSiNoiseMap<HGCSiliconDetId> scal_;
  HGCalSiNoiseMap<HFNoseDetId> scalHFNose_;

  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<DFr> > myFEelectronics_;

  //bunch time
  double bxTime_;

  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

  //if set to true, threshold will be computed based on the expected meap peak/2
  bool thresholdFollowsMIP_;

  // Identify the detector components, i.e. DetIds, that will be managed by
  // this digitizer. This information will be used to fetch the correct
  // geometry and the full list of detids for which a digitization is
  // requested.
  DetId::Detector det_;

  // Identify the subdetector components that will be managed by this
  // digitizer. This information will be used to fetch the correct geometry and
  // the full list of detids for which a digitization is requested.
  ForwardSubdetector subdet_;

  // New NoiseArray Parameters

  const double NoiseMean_, NoiseStd_;
  static const size_t NoiseArrayLength_ = 200000;
  static const size_t samplesize_ = 15;
  std::array<std::array<double, samplesize_>, NoiseArrayLength_> GaussianNoiseArray_;
  bool RandNoiseGenerationFlag_;
  // A parameter configurable from python configuration to decide which noise generation model to use
  bool NoiseGeneration_Method_;
};

#endif
