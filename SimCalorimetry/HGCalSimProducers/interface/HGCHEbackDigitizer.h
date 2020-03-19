#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"

class HGCHEbackDigitizer : public HGCDigitizerBase<HGCalDataFrame> {
public:
  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                    hgc::HGCSimHitDataAccumulator& simData,
                    const CaloSubdetectorGeometry* theGeom,
                    const std::unordered_set<DetId>& validIds,
                    uint32_t digitizationType,
                    CLHEP::HepRandomEngine* engine) override;
  ~HGCHEbackDigitizer() override;

private:
  //calice-like digitization parameters
  uint32_t algo_;
  bool scaleByTileArea_, scaleBySipmArea_, scaleByDose_, thresholdFollowsMIP_;
  float keV2MIP_, noise_MIP_;
  float nPEperMIP_, nTotalPE_, xTalk_, sdPixels_;
  std::string doseMapFile_, sipmMapFile_;
  HGCalSciNoiseMap scal_;

  void runEmptyDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                         hgc::HGCSimHitDataAccumulator& simData,
                         const CaloSubdetectorGeometry* theGeom,
                         const std::unordered_set<DetId>& validIds,
                         CLHEP::HepRandomEngine* engine);

  void runRealisticDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                             hgc::HGCSimHitDataAccumulator& simData,
                             const CaloSubdetectorGeometry* theGeom,
                             const std::unordered_set<DetId>& validIds,
                             CLHEP::HepRandomEngine* engine);

  void runCaliceLikeDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                              hgc::HGCSimHitDataAccumulator& simData,
                              const CaloSubdetectorGeometry* theGeom,
                              const std::unordered_set<DetId>& validIds,
                              CLHEP::HepRandomEngine* engine);
};

#endif
