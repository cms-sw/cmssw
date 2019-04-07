#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"


class HGCHEbackSignalScaler
{
  public:

    struct DoseParameters {
      DoseParameters(): a_(0.), b_(0.), c_(0.), d_(0.), e_(0.), f_(0.) {}
      float a_, b_, c_, d_, e_, f_;
    };

    HGCHEbackSignalScaler() {};
    ~HGCHEbackSignalScaler() {};

    void setGeometry(const CaloSubdetectorGeometry*);
    void setDoseMap(const std::string&);

    float scaleByArea(const HGCScintillatorDetId&);
    std::pair<float, float> scaleByDose(const HGCScintillatorDetId&);
    double getDoseValue(const int, const float);
    double getFluenceValue(const int, const float);
    float computeEdge(const HGCScintillatorDetId&);
    float computeRadius(const HGCScintillatorDetId&);

  private:
    std::map<int, DoseParameters> readDosePars(const std::string&);

    const HGCalGeometry* hgcalGeom_;
    std::map<int, DoseParameters> doseMap_;

    bool verbose_ = false;

};



class HGCHEbackDigitizer : public HGCDigitizerBase<HGCalDataFrame>
{
 public:

  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,
		    const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
		    uint32_t digitizationType, CLHEP::HepRandomEngine* engine) override;
  ~HGCHEbackDigitizer() override;

 private:

  //calice-like digitization parameters
  uint32_t algo_;
  bool scaleByArea_, scaleByDose_, calibDigis_;
  float keV2MIP_, noise_MIP_;
  float nPEperMIP_, nTotalPE_, xTalk_, sdPixels_;
  std::string doseMapFile_;
  HGCHEbackSignalScaler scal_;

  void runEmptyDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,
                         const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
                         CLHEP::HepRandomEngine* engine);

  void runRealisticDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,
                             const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
                             CLHEP::HepRandomEngine* engine);

  void runCaliceLikeDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,
			      const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
			      CLHEP::HepRandomEngine* engine);
};

#endif
