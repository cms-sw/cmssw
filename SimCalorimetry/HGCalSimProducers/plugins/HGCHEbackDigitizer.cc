#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "vdt/vdtMath.h"

using namespace hgc_digi;
using namespace hgc_digi_utils;

class HGCHEbackDigitizer : public HGCDigitizerBase {
public:
  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                    hgc::HGCSimHitDataAccumulator& simData,
                    const CaloSubdetectorGeometry* theGeom,
                    const std::unordered_set<DetId>& validIds,
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

HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  edm::ParameterSet cfg = ps.getParameter<edm::ParameterSet>("digiCfg");
  algo_ = cfg.getParameter<uint32_t>("algo");
  scaleByTileArea_ = cfg.getParameter<bool>("scaleByTileArea");
  scaleBySipmArea_ = cfg.getParameter<bool>("scaleBySipmArea");
  sipmMapFile_ = cfg.getParameter<std::string>("sipmMap");
  scaleByDose_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<bool>("scaleByDose");
  unsigned int scaleByDoseAlgo = cfg.getParameter<edm::ParameterSet>("noise").getParameter<uint32_t>("scaleByDoseAlgo");
  scaleByDoseFactor_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("scaleByDoseFactor");
  doseMapFile_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<std::string>("doseMap");
  noise_MIP_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("noise_MIP");
  thresholdFollowsMIP_ = cfg.getParameter<bool>("thresholdFollowsMIP");
  keV2MIP_ = cfg.getParameter<double>("keV2MIP");
  this->keV2fC_ = 1.0;  //keV2MIP_; // hack for HEB
  this->det_ = DetId::HGCalHSc;
  nPEperMIP_ = cfg.getParameter<double>("nPEperMIP");
  nTotalPE_ = cfg.getParameter<double>("nTotalPE");
  xTalk_ = cfg.getParameter<double>("xTalk");
  sdPixels_ = cfg.getParameter<double>("sdPixels");

  scal_.setDoseMap(doseMapFile_, scaleByDoseAlgo);
  scal_.setFluenceScaleFactor(scaleByDoseFactor_);
  scal_.setSipmMap(sipmMapFile_);
}

//
void HGCHEbackDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                      HGCSimHitDataAccumulator& simData,
                                      const CaloSubdetectorGeometry* theGeom,
                                      const std::unordered_set<DetId>& validIds,
                                      CLHEP::HepRandomEngine* engine) {
  if (algo_ == 0)
    runEmptyDigitizer(digiColl, simData, theGeom, validIds, engine);
  else if (algo_ == 1)
    runCaliceLikeDigitizer(digiColl, simData, theGeom, validIds, engine);
  else if (algo_ == 2)
    runRealisticDigitizer(digiColl, simData, theGeom, validIds, engine);
}

void HGCHEbackDigitizer::runEmptyDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                           HGCSimHitDataAccumulator& simData,
                                           const CaloSubdetectorGeometry* theGeom,
                                           const std::unordered_set<DetId>& validIds,
                                           CLHEP::HepRandomEngine* engine) {
  HGCSimHitData chargeColl, toa;
  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f);  //accumulated energy
  zeroData.hit_info[1].fill(0.f);  //time-of-flight

  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);

    for (size_t i = 0; i < cell.hit_info[0].size(); ++i) {
      //convert total energy keV->MIP, since converted to keV in accumulator
      const float totalIniMIPs(cell.hit_info[0][i] * keV2MIP_);

      //store
      chargeColl[i] = totalIniMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, engine);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
}

void HGCHEbackDigitizer::runRealisticDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                               HGCSimHitDataAccumulator& simData,
                                               const CaloSubdetectorGeometry* theGeom,
                                               const std::unordered_set<DetId>& validIds,
                                               CLHEP::HepRandomEngine* engine) {
  //switch to true if you want to print some details
  constexpr bool debug(false);

  HGCSimHitData chargeColl, toa;
  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f);  //accumulated energy
  zeroData.hit_info[1].fill(0.f);  //time-of-flight

  // needed to compute the radiation and geometry scale factors
  scal_.setGeometry(theGeom);

  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);

    float scaledPePerMip = nPEperMIP_;           //needed to scale according to tile geometry
    float tunedNoise = nPEperMIP_ * noise_MIP_;  //flat noise case
    float sipmFactor = 1.;                       //standard 2 mm^2 sipm

    if (id.det() == DetId::HGCalHSc)  //skip those geometries that have HE used as BH
    {
      double radius(0);
      if (scaleByTileArea_ or scaleByDose_ or scaleBySipmArea_)
        radius = scal_.computeRadius(id);

      //take into account the tile size
      if (scaleByTileArea_)
        scaledPePerMip *= scal_.scaleByTileArea(id, radius);

      //take into account the darkening of the scintillator and SiPM dark current
      if (scaleByDose_) {
        auto dosePair = scal_.scaleByDose(id, radius);
        scaledPePerMip *= dosePair.first;
        tunedNoise = dosePair.second;
      }

      //take into account the sipm size
      if (scaleBySipmArea_) {
        sipmFactor = scal_.scaleBySipmArea(id, radius);
        scaledPePerMip *= sipmFactor;
        tunedNoise *= sqrt(sipmFactor);
      }
    }

    //set mean for poissonian noise
    float meanN = std::pow(tunedNoise, 2);

    for (size_t i = 0; i < cell.hit_info[0].size(); ++i) {
      //convert total energy keV->MIP, since converted to keV in accumulator
      float totalIniMIPs(cell.hit_info[0][i] * keV2MIP_);

      //generate the number of photo-electrons from the energy deposit
      const uint32_t npeS = std::floor(CLHEP::RandPoissonQ::shoot(engine, totalIniMIPs * scaledPePerMip) + 0.5);

      //generate the noise associated to the dark current
      const uint32_t npeN = std::floor(CLHEP::RandPoissonQ::shoot(engine, meanN) + 0.5);

      //total number of pe from signal + noise  (not subtracting pedestal)
      const uint32_t npe = npeS + npeN;

      //take into account SiPM saturation
      float nTotalPixels = nTotalPE_ * sipmFactor;
      const float x = vdt::fast_expf(-((float)npe) / nTotalPixels);
      uint32_t nPixel(0);
      if (xTalk_ * x != 1)
        nPixel = (uint32_t)std::max(nTotalPixels * (1.f - x) / (1.f - xTalk_ * x), 0.f);

      //take into account the gain fluctuations of each pixel
      //const float nPixelTot = nPixel + sqrt(nPixel) * CLHEP::RandGaussQ::shoot(engine, 0., 0.05); //FDG: just a note for now, par to be defined

      //scale to calibrated response depending on the thresholdFollowsMIP_ flag
      float totalMIPs = thresholdFollowsMIP_ ? std::max((npe - meanN), 0.f) / nPEperMIP_ : nPixel / nPEperMIP_;

      if (debug && totalIniMIPs > 0) {
        LogDebug("HGCHEbackDigitizer") << "npeS: " << npeS << " npeN: " << npeN << " npe: " << npe
                                       << " meanN: " << meanN << " noise_MIP_: " << noise_MIP_
                                       << " nPEperMIP_: " << nPEperMIP_ << " scaledPePerMip: " << scaledPePerMip
                                       << " nPixel: " << nPixel;
        LogDebug("HGCHEbackDigitizer") << "totalIniMIPs: " << totalIniMIPs << " totalMIPs: " << totalMIPs << std::endl;
      }

      //store charge
      chargeColl[i] = totalMIPs;

      //update time of arrival
      toa[i] = cell.hit_info[1][i];
      if (myFEelectronics_->toaMode() == HGCFEElectronics<HGCalDataFrame>::WEIGHTEDBYE && totalIniMIPs > 0)
        toa[i] = cell.hit_info[1][i] / totalIniMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    float adcThr = this->myFEelectronics_->getADCThreshold();  //this is in MIPs
    float adcLsb = this->myFEelectronics_->getADClsb();
    uint32_t thrADC(thresholdFollowsMIP_ ? std::floor(adcThr / adcLsb * scaledPePerMip / nPEperMIP_)
                                         : std::floor(adcThr / adcLsb));

    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, engine, thrADC);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
}

//
void HGCHEbackDigitizer::runCaliceLikeDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                                HGCSimHitDataAccumulator& simData,
                                                const CaloSubdetectorGeometry* theGeom,
                                                const std::unordered_set<DetId>& validIds,
                                                CLHEP::HepRandomEngine* engine) {
  //switch to true if you want to print some details
  constexpr bool debug(false);

  HGCSimHitData chargeColl, toa;

  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f);  //accumulated energy
  zeroData.hit_info[1].fill(0.f);  //time-of-flight

  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);

    for (size_t i = 0; i < cell.hit_info[0].size(); ++i) {
      //convert total energy keV->MIP, since converted to keV in accumulator
      const float totalIniMIPs(cell.hit_info[0][i] * keV2MIP_);

      //generate random number of photon electrons
      const uint32_t npe = std::floor(CLHEP::RandPoissonQ::shoot(engine, totalIniMIPs * nPEperMIP_));

      //number of pixels
      const float x = vdt::fast_expf(-((float)npe) / nTotalPE_);
      uint32_t nPixel(0);
      if (xTalk_ * x != 1)
        nPixel = (uint32_t)std::max(nTotalPE_ * (1.f - x) / (1.f - xTalk_ * x), 0.f);

      //update signal
      if (sdPixels_ != 0)
        nPixel = (uint32_t)std::max(CLHEP::RandGaussQ::shoot(engine, (double)nPixel, sdPixels_), 0.);

      //convert to MIP again and saturate
      float totalMIPs(0.f), xtalk = 0.f;
      const float peDiff = nTotalPE_ - (float)nPixel;
      if (peDiff != 0.f) {
        xtalk = (nTotalPE_ - xTalk_ * ((float)nPixel)) / peDiff;
        if (xtalk > 0.f && nPEperMIP_ != 0.f)
          totalMIPs = (nTotalPE_ / nPEperMIP_) * vdt::fast_logf(xtalk);
      }

      //add noise (in MIPs)
      chargeColl[i] = totalMIPs;
      if (noise_MIP_ != 0)
        chargeColl[i] += std::max(CLHEP::RandGaussQ::shoot(engine, 0., noise_MIP_), 0.);
      if (debug && cell.hit_info[0][i] > 0)
        edm::LogVerbatim("HGCDigitizer") << "[runCaliceLikeDigitizer] xtalk=" << xtalk << " En=" << cell.hit_info[0][i]
                                         << " keV -> " << totalIniMIPs << " raw-MIPs -> " << chargeColl[i]
                                         << " digi-MIPs";
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, engine);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer() {}

DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCHEbackDigitizer, "HGCHEbackDigitizer");
