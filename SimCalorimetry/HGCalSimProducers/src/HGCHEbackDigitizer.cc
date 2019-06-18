#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "vdt/vdtMath.h"

using namespace hgc_digi;
using namespace hgc_digi_utils;

void HGCHEbackSignalScaler::setDoseMap(const std::string& fullpath) { doseMap_ = readDosePars(fullpath); }

void HGCHEbackSignalScaler::setGeometry(const CaloSubdetectorGeometry* geom) {
  hgcalGeom_ = static_cast<const HGCalGeometry*>(geom);
}

std::map<int, HGCHEbackSignalScaler::DoseParameters> HGCHEbackSignalScaler::readDosePars(const std::string& fullpath) {
  std::map<int, DoseParameters> result;

  //no dose file means no aging
  if (fullpath.empty())
    return result;

  edm::FileInPath fp(fullpath);
  std::ifstream infile(fp.fullPath());
  if (!infile.is_open()) {
    throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
  }
  std::string line;
  while (getline(infile, line)) {
    int layer;
    DoseParameters dosePars;

    //space-separated
    std::stringstream linestream(line);
    linestream >> layer >> dosePars.a_ >> dosePars.b_ >> dosePars.c_ >> dosePars.d_ >> dosePars.e_ >> dosePars.f_ >>
        dosePars.g_ >> dosePars.h_ >> dosePars.i_ >> dosePars.j_;

    result[layer] = dosePars;
  }
  return result;
}

double HGCHEbackSignalScaler::getDoseValue(const int layer, const std::array<double, 8>& radius) {
  double cellDose = std::pow(10,
                             doseMap_[layer].a_ + doseMap_[layer].b_ * radius[4] + doseMap_[layer].c_ * radius[5] +
                                 doseMap_[layer].d_ * radius[6] + doseMap_[layer].e_ * radius[7]);  //dose in grey
  return cellDose * greyToKrad_;                                                                    //convert to kRad
}

double HGCHEbackSignalScaler::getFluenceValue(const int layer, const std::array<double, 8>& radius) {
  double cellFluence = std::pow(10,
                                doseMap_[layer].f_ + doseMap_[layer].g_ * radius[0] + doseMap_[layer].h_ * radius[1] +
                                    doseMap_[layer].i_ * radius[2] + doseMap_[layer].j_ * radius[3]);  //dose in grey
  return cellFluence;
}

std::pair<float, float> HGCHEbackSignalScaler::scaleByDose(const HGCScintillatorDetId& cellId,
                                                           const std::array<double, 8>& radius) {
  if (doseMap_.empty())
    return std::make_pair(1., 0.);

  int layer = cellId.layer();
  double cellDose = getDoseValue(layer, radius);  //in kRad
  constexpr double expofactor = 1. / 199.6;
  double scaleFactor = std::exp(-std::pow(cellDose, 0.65) * expofactor);

  double cellFluence = getFluenceValue(layer, radius);  //in 1-Mev-equivalent neutrons per cm2

  constexpr double factor = 2. / (2 * 1e13);  //SiPM area = 2mm^2
  double noise = 2.18 * sqrt(cellFluence * factor);

  if (verbose_) {
    LogDebug("HGCHEbackSignalScaler") << "HGCHEbackSignalScaler::scaleByDose - Dose, scaleFactor, fluence, noise: "
                                      << cellDose << " " << scaleFactor << " " << cellFluence << " " << noise;

    LogDebug("HGCHEbackSignalScaler") << "HGCHEbackSignalScaler::setDoseMap - layer, a, b, c, d, e, f: " << layer << " "
                                      << doseMap_[layer].a_ << " " << doseMap_[layer].b_ << " " << doseMap_[layer].c_
                                      << " " << doseMap_[layer].d_ << " " << doseMap_[layer].e_ << " "
                                      << doseMap_[layer].f_;
  }

  return std::make_pair(scaleFactor, noise);
}

float HGCHEbackSignalScaler::scaleByArea(const HGCScintillatorDetId& cellId, const std::array<double, 8>& radius) {
  float edge;
  if (cellId.type() == 0) {
    constexpr double factor = 2 * M_PI * 1. / 360.;
    edge = radius[0] * factor;  //1 degree
  } else {
    constexpr double factor = 2 * M_PI * 1. / 288.;
    edge = radius[0] * factor;  //1.25 degrees
  }

  float scaleFactor = refEdge_ / edge;  //assume reference 3cm of edge

  if (verbose_) {
    LogDebug("HGCHEbackSignalScaler") << "HGCHEbackSignalScaler::scaleByArea - Type, layer, edge, radius, SF: "
                                      << cellId.type() << " " << cellId.layer() << " " << edge << " " << radius[0]
                                      << " " << scaleFactor << std::endl;
  }

  return scaleFactor;
}

std::array<double, 8> HGCHEbackSignalScaler::computeRadius(const HGCScintillatorDetId& cellId) {
  GlobalPoint global = hgcalGeom_->getPosition(cellId);

  double radius2 = std::pow(global.x(), 2) + std::pow(global.y(), 2);  //in cm
  double radius4 = std::pow(radius2, 2);
  double radius = sqrt(radius2);
  double radius3 = radius2 * radius;

  double radius_m100 = radius - 100;
  double radius_m100_2 = std::pow(radius_m100, 2);
  double radius_m100_3 = radius_m100_2 * radius_m100;
  double radius_m100_4 = std::pow(radius_m100_2, 2);

  std::array<double, 8> radii{
      {radius, radius2, radius3, radius4, radius_m100, radius_m100_2, radius_m100_3, radius_m100_4}};
  return radii;
}

//--- the actual digitizer --------------------------------------------------------------------------------------------------
HGCHEbackDigitizer::HGCHEbackDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  edm::ParameterSet cfg = ps.getParameter<edm::ParameterSet>("digiCfg");
  algo_ = cfg.getParameter<uint32_t>("algo");
  scaleByArea_ = cfg.getParameter<bool>("scaleByArea");
  scaleByDose_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<bool>("scaleByDose");
  doseMapFile_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<std::string>("doseMap");
  noise_MIP_ = cfg.getParameter<edm::ParameterSet>("noise").getParameter<double>("noise_MIP");
  calibDigis_ = cfg.getParameter<bool>("calibDigis");
  keV2MIP_ = cfg.getParameter<double>("keV2MIP");
  this->keV2fC_ = 1.0;  //keV2MIP_; // hack for HEB
  nPEperMIP_ = cfg.getParameter<double>("nPEperMIP");
  nTotalPE_ = cfg.getParameter<double>("nTotalPE");
  xTalk_ = cfg.getParameter<double>("xTalk");
  sdPixels_ = cfg.getParameter<double>("sdPixels");

  scal_.setDoseMap(doseMapFile_);
}

//
void HGCHEbackDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                      HGCSimHitDataAccumulator& simData,
                                      const CaloSubdetectorGeometry* theGeom,
                                      const std::unordered_set<DetId>& validIds,
                                      uint32_t digitizationType,
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
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, 1, engine);

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

    if (id.det() == DetId::HGCalHSc)  //skip those geometries that have HE used as BH
    {
      std::array<double, 8> radius;
      if (scaleByArea_ or scaleByDose_)
        radius = scal_.computeRadius(id);

      if (scaleByArea_)
        scaledPePerMip *= scal_.scaleByArea(id, radius);

      //take into account the darkening of the scintillator and SiPM dark current
      if (scaleByDose_) {
        auto dosePair = scal_.scaleByDose(id, radius);
        scaledPePerMip *= dosePair.first;
        tunedNoise = dosePair.second;
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
      const float x = vdt::fast_expf(-((float)npe) / nTotalPE_);
      uint32_t nPixel(0);
      if (xTalk_ * x != 1)
        nPixel = (uint32_t)std::max(nTotalPE_ * (1.f - x) / (1.f - xTalk_ * x), 0.f);

      //take into account the gain fluctuations of each pixel
      //const float nPixelTot = nPixel + sqrt(nPixel) * CLHEP::RandGaussQ::shoot(engine, 0., 0.05); //FDG: just a note for now, par to be defined

      //scale to calibrated response depending on the calibDigis_ flag
      float totalMIPs = calibDigis_ ? (float)npe / scaledPePerMip : nPixel / nPEperMIP_;

      if (debug && totalIniMIPs > 0) {
        LogDebug("HGCHEbackDigitizer") << "npeS: " << npeS << " npeN: " << npeN << " npe: " << npe
                                       << " meanN: " << meanN << " noise_MIP_: " << noise_MIP_
                                       << " nPEperMIP_: " << nPEperMIP_ << " scaledPePerMip: " << scaledPePerMip
                                       << " nPixel: " << nPixel;
        LogDebug("HGCHEbackDigitizer") << "totalIniMIPs: " << totalIniMIPs << " totalMIPs: " << totalMIPs << std::endl;
      }

      //store
      chargeColl[i] = totalMIPs;
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, 1, engine);

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
	edm::LogVerbatim("HGCDigitizer") 
	  << "[runCaliceLikeDigitizer] xtalk=" << xtalk << " En=" << cell.hit_info[0][i] << " keV -> "
	  << totalIniMIPs << " raw-MIPs -> " << chargeColl[i] << " digi-MIPs";
    }

    //init a new data frame and run shaper
    HGCalDataFrame newDataFrame(id);
    this->myFEelectronics_->runShaper(newDataFrame, chargeColl, toa, 1, engine);

    //prepare the output
    this->updateOutput(digiColl, newDataFrame);
  }
}

//
HGCHEbackDigitizer::~HGCHEbackDigitizer() {}
