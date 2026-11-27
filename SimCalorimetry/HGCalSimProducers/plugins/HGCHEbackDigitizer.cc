#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandBinomial.h"
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
  bool scaleByDose_, thresholdFollowsMIP_;
  float keV2MIP_, noise_MIP_, gainValue_;
  float nPEperMIP_, nTotalPE_, xTalk_, sigma_;
  std::string doseMapFile_;
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

  void runRealisticSciDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
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
  scaleByDose_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<bool>("scaleByDose");
  unsigned int scaleByDoseAlgo = cfg.getParameter<edm::ParameterSet>("noise").getParameter<uint32_t>("scaleByDoseAlgo");
  scaleByDoseFactor_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("scaleByDoseFactor");
  doseMapFile_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<std::string>("doseMap");
  noise_MIP_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("noise_MIP");
  double refIdark = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("referenceIdark");
  xTalk_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("referenceXtalk");
  thresholdFollowsMIP_ = cfg.getParameter<bool>("thresholdFollowsMIP");
  keV2MIP_ = cfg.getParameter<double>("keV2MIP");
  this->keV2fC_ = 1.0;  //keV2MIP_; // hack for HEB
  this->det_ = DetId::HGCalHSc;
  nPEperMIP_ = cfg.getParameter<double>("nPEperMIP");
  nTotalPE_ = cfg.getParameter<double>("nTotalPE");
  gainValue_ = cfg.getParameter<double>("gain");  // Constant SiPM gain in units of ADC for the calculation of digis
  sigma_ = cfg.getParameter<double>("sigma");

  scal_.setDoseMap(doseMapFile_, scaleByDoseAlgo);
  scal_.setReferenceDarkCurrent(refIdark);
  scal_.setFluenceScaleFactor(scaleByDoseFactor_);
  scal_.setReferenceCrossTalk(xTalk_);

  //but the TDC scale needs to be updated to use pe instead of MIP units      // TDC still to be properly implemented for the updated digitiser (TODO)
  if (scaleByDose_)
    this->myFEelectronics_->setTDCfsc(2 * scal_.getNPeInSiPM());
}
// ----------------------

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
  else if (algo_ == 3)
    runRealisticSciDigitizer(digiColl, simData, theGeom, validIds, engine);
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

  //vanilla reference values are indepenent of the ids and were set by
  //configuration in the python - no need to recomput them every time
  //in the digitization loop
  float scaledPePerMip = nPEperMIP_;                                //needed to scale according to tile geometry
  float tunedNoise = nPEperMIP_ * noise_MIP_;                       //flat noise case
  float vanillaADCThr = this->myFEelectronics_->getADCThreshold();  //vanilla thrs  in MIPs
  float adcLsb(this->myFEelectronics_->getADClsb());
  float maxADC(-1);  //vanilla will rely on what has been configured by default
  uint32_t thrADC(thresholdFollowsMIP_ ? std::floor(vanillaADCThr / adcLsb * scaledPePerMip / nPEperMIP_)
                                       : std::floor(vanillaADCThr / adcLsb));
  float nTotalPixels(nTotalPE_);
  float xTalk(xTalk_);
  int gainIdx(0);

  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);

    //in case realistic scenario (noise, fluence, dose, sipm/tile area) are to be used
    //we update vanilla values with the realistic ones
    if (id.det() == DetId::HGCalHSc && scaleByDose_) {
      HGCScintillatorDetId scId(id.rawId());
      auto opChar = scal_.scaleByDose(scId);
      scaledPePerMip = opChar.s;
      tunedNoise = opChar.n;
      gainIdx = opChar.gain;
      thrADC = opChar.thrADC;
      adcLsb = scal_.getLSBPerGain()[gainIdx];
      maxADC = scal_.getMaxADCPerGain()[gainIdx] - 1e-6;
      nTotalPixels = opChar.ntotalPE;
      xTalk = opChar.xtalk;
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
      uint32_t nPixel(npe);
      if (xTalk >= 0) {
        const float x = vdt::fast_expf(-((float)npe) / nTotalPixels);
        if (xTalk_ * x != 1)
          nPixel = (uint32_t)std::max(nTotalPixels * (1.f - x) / (1.f - xTalk_ * x), 0.f);
      }

      //take into account the gain fluctuations of each pixel
      //FDG: just a note for now, par to be defined
      //const float nPixelTot = nPixel + sqrt(nPixel) * CLHEP::RandGaussQ::shoot(engine, 0., 0.05);

      //realistic behavior: subtract the pedestal
      //Note: for now the saturation effects are ignored...
      if (scaleByDose_) {
        float pedestal(meanN);
        if (scal_.ignoreAutoPedestalSubtraction())
          pedestal = 0.f;
        chargeColl[i] = std::max(nPixel - pedestal, 0.f);
      }
      //vanilla simulation: scale back to MIP units... and to calibrated response depending on the thresholdFollowsMIP_ flag
      else {
        float totalMIPs = thresholdFollowsMIP_ ? std::max((nPixel - meanN), 0.f) / nPEperMIP_ : nPixel / nPEperMIP_;

        if (debug && totalIniMIPs > 0) {
          LogDebug("HGCHEbackDigitizer") << "npeS: " << npeS << " npeN: " << npeN << " npe: " << npe
                                         << " meanN: " << meanN << " noise_MIP_: " << noise_MIP_
                                         << " nPEperMIP_: " << nPEperMIP_ << " scaledPePerMip: " << scaledPePerMip
                                         << " nPixel: " << nPixel;
          LogDebug("HGCHEbackDigitizer") << "totalIniMIPs: " << totalIniMIPs << " totalMIPs: " << totalMIPs
                                         << std::endl;
        }

        //store charge
        chargeColl[i] = totalMIPs;
      }

      //update time of arrival
      toa[i] = cell.hit_info[1][i];
      if (myFEelectronics_->toaMode() == HGCFEElectronics<HGCalDataFrame>::WEIGHTEDBYE && totalIniMIPs > 0)
        toa[i] = cell.hit_info[1][i] / totalIniMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, engine, thrADC, adcLsb, gainIdx, maxADC);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
}

void HGCHEbackDigitizer::runRealisticSciDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                                  HGCSimHitDataAccumulator& simData,
                                                  const CaloSubdetectorGeometry* theGeom,
                                                  const std::unordered_set<DetId>& validIds,
                                                  CLHEP::HepRandomEngine* engine) {
  //switch to true if you want to print some details
  constexpr bool debug(false);
  constexpr bool doSaturation(true);  // flag to perform emulation of saturation of SiPM pixels
  constexpr bool doSmearing(true);    // flag to perform emulation of statistical fluctiations of n of fired pixels
  constexpr bool doNoise(false);      // flag to add noise

  HGCSimHitData chargeColl, toa;
  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f);  //accumulated energy
  zeroData.hit_info[1].fill(0.f);  //time-of-flight

  // needed to compute the radiation and geometry scale factors
  scal_.setGeometry(theGeom);  // unused at the moment, untested (10/2026)

  float scaledLYPerMip = -98.8;
  float tunedNoise = nPEperMIP_ * noise_MIP_;  // flat noise case // UPD: outdated at the mement (10/2026)
  float vanillaADCThr = this->myFEelectronics_->getADCThreshold();  // vanilla thrs in MIPs

  float maxADC(-1);  //vanilla will rely on what has been configured by default
  float gainValue(gainValue_);

  // threshold calcuilated based on the expected ADC for ope MIP * the vanilla threshold in MIPs:
  uint32_t thrADC(thresholdFollowsMIP_ ? std::floor(vanillaADCThr * scaledLYPerMip * gainValue)
                                       : std::floor(vanillaADCThr));
  float nTotalPx(nTotalPE_);
  //float xTalk(xTalk_);
  int gainIdx(0);  // UPD: unused, incorrect assumption about channel equalisation

  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);
    float simHitADC(0.f);
    int nPixels_sat = -99;
    int nPixels_sat_smear = -88;
    double radius;  //, x, y, z;
    //double tileE, tileA;
    float pedestal(150);  // value of the expected position of the pedestal determined for the runs

    HGCScintillatorDetId scId(id.rawId());
    radius = static_cast<const HGCalGeometry*>(theGeom)->getPosition(scId).perp();
    double edge(pow(static_cast<const HGCalGeometry*>(theGeom)->getArea(scId), 0.5));
    auto opChar = scal_.scaleByDoseLy(scId, radius, edge, id.rawId(), sigma_);  // using new for digitiser update
    scaledLYPerMip = opChar.lySF;
    tunedNoise = opChar.n / nPEperMIP_ * scaledLYPerMip;
    gainIdx = opChar.gain;
    thrADC = std::floor(vanillaADCThr * scaledLYPerMip * gainValue);
    float adcLsb = 1;

    //set mean for poissonian noise
    float meanN = std::pow(tunedNoise, 2);
    if (scaleByDose_ && doNoise) {
      pedestal = meanN;
      if (scal_.ignoreAutoPedestalSubtraction())
        pedestal = 0.f;
    }

    maxADC = 1024 - pedestal;

    for (size_t i = 0; i < cell.hit_info[0].size(); ++i) {
      //convert total energy keV->MIP, since converted to keV in accumulator
      float totalIniMIPs(cell.hit_info[0][i] * keV2MIP_);

      // generate the mean number of photo-electrons from the energy deposit
      int nPixels = std::floor(totalIniMIPs * scaledLYPerMip);
      // apply SiPM saturation function
      if (doSaturation) {
        nPixels_sat = std::floor(nTotalPx * (1 - vdt::fast_expf(-nPixels / nTotalPx)));
        // Binomial smearing to emulate statistical fluctuations for fired pixels
        if (doSmearing) {
          float prob = nPixels_sat / nTotalPx;
          nPixels_sat_smear = std::floor(CLHEP::RandBinomial::shoot(engine, nTotalPx, prob));
          if (doNoise) {
            //generate the noise associated to the dark current
            const uint32_t npeN = std::floor(CLHEP::RandPoissonQ::shoot(engine, meanN) + 0.5);
            nPixels_sat_smear += std::max(0.f, npeN - pedestal);
          }
          if (nPixels_sat_smear > 0) {
            simHitADC = nPixels_sat_smear * gainValue;
          } else {
            simHitADC = 0.;
          }
        } else {
          simHitADC = nPixels_sat * gainValue;
        }
      } else {
        simHitADC = nPixels * gainValue;
      }
      //store ADC
      chargeColl[i] = simHitADC;

      if (debug && cell.hit_info[0][i] > 0)
        edm::LogVerbatim("HGCDigitizer")
            << "Realistic Sci Digitizer\n"
            << "keV " << cell.hit_info[0][i] << "\ntotalIniMIPs " << totalIniMIPs << "\nADC(9mm2 ConvG=2 OV=4) "
            << chargeColl[i]
            //<< "ADCthresh * ly = " << vanillaADCThr* scaledPePerMip
            << "\nvanillaADCThr = " << vanillaADCThr << "\n ADC thr " << thrADC << "\nnPixels " << nPixels
            << "\nnPixels_sat " << nPixels_sat << "\nnTotalPx " << nTotalPx << "\nnPixels_sat_smear "
            << nPixels_sat_smear << "\ngainValue " << gainValue << " cellId.granularity() "
            << scId.granularity()
            // << "\nradius "<< radius << " X: " << x << " Y: " << y << " Z: " << z
            // << "\nTileAreaFactor " << tileE << " 3/tileE = " << 3.0/ tileE << " Tile Area: " << tileA
            // << "\nHGCalHSc "<< (id.det() == DetId::HGCalHSc)
            // << "\nscaledPePerMip "<< scaledPePerMip
            // << "\ngainIdx " << gainIdx
            << "\n\nid.rawId() " << id.rawId() << "\nSigma " << sigma_ << "\nscaledLYPerMip " << scaledLYPerMip;

      //update time of arrival
      toa[i] = cell.hit_info[1][i];
      if (myFEelectronics_->toaMode() == HGCFEElectronics<HGCalDataFrame>::WEIGHTEDBYE && totalIniMIPs > 0)
        toa[i] = cell.hit_info[1][i] / totalIniMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, engine, thrADC, adcLsb, gainIdx, maxADC);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
  //outfile.close();
}
/// runRealisticSciDigitizer

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
      if (sigma_ != 0)
        nPixel = (uint32_t)std::max(CLHEP::RandGaussQ::shoot(engine, (double)nPixel, sigma_), 0.);

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
