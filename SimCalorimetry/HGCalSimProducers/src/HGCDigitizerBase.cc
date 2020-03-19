#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

using namespace hgc_digi;
using namespace hgc_digi_utils;

template <class DFr>
HGCDigitizerBase<DFr>::HGCDigitizerBase(const edm::ParameterSet& ps)
    : scaleByDose_(false), NoiseMean_(0.0), NoiseStd_(1.0) {
  bxTime_ = ps.getParameter<double>("bxTime");
  myCfg_ = ps.getParameter<edm::ParameterSet>("digiCfg");
  NoiseGeneration_Method_ = ps.getParameter<bool>("NoiseGeneration_Method");
  doTimeSamples_ = myCfg_.getParameter<bool>("doTimeSamples");
  thresholdFollowsMIP_ = myCfg_.getParameter<bool>("thresholdFollowsMIP");
  if (myCfg_.exists("keV2fC"))
    keV2fC_ = myCfg_.getParameter<double>("keV2fC");
  else
    keV2fC_ = 1.0;

  if (myCfg_.existsAs<edm::ParameterSet>("chargeCollectionEfficiencies")) {
    cce_ = myCfg_.getParameter<edm::ParameterSet>("chargeCollectionEfficiencies")
               .template getParameter<std::vector<double>>("values");
  }

  if (myCfg_.existsAs<double>("noise_fC")) {
    noise_fC_.reserve(1);
    noise_fC_.push_back(myCfg_.getParameter<double>("noise_fC"));
  } else if (myCfg_.existsAs<std::vector<double>>("noise_fC")) {
    const auto& noises = myCfg_.getParameter<std::vector<double>>("noise_fC");
    noise_fC_ = std::vector<float>(noises.begin(), noises.end());
  } else if (myCfg_.existsAs<edm::ParameterSet>("noise_fC")) {
    const auto& noises =
        myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<std::vector<double>>("values");
    noise_fC_ = std::vector<float>(noises.begin(), noises.end());
    scaleByDose_ = myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<bool>("scaleByDose");
    int scaleByDoseAlgo =
        myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<uint32_t>("scaleByDoseAlgo");
    doseMapFile_ = myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<std::string>("doseMap");
    scal_.setDoseMap(doseMapFile_, scaleByDoseAlgo);
  } else {
    noise_fC_.resize(1, 1.f);
  }
  if (myCfg_.existsAs<edm::ParameterSet>("ileakParam")) {
    scal_.setIleakParam(
        myCfg_.getParameter<edm::ParameterSet>("ileakParam").template getParameter<std::vector<double>>("ileakParam"));
  }
  if (myCfg_.existsAs<edm::ParameterSet>("cceParams")) {
    scal_.setCceParam(
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamFine"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThin"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThick"));
  }
  edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
  myFEelectronics_ = std::unique_ptr<HGCFEElectronics<DFr>>(new HGCFEElectronics<DFr>(feCfg));
  myFEelectronics_->SetNoiseValues(noise_fC_);
  RandNoiseGenerationFlag_ = 0;
}

template <class DFr>
void HGCDigitizerBase<DFr>::GenerateGaussianNoise(CLHEP::HepRandomEngine* engine,
                                                  const double NoiseMean,
                                                  const double NoiseStd) {
  for (size_t i = 0; i < NoiseArrayLength_; i++) {
    for (size_t j = 0; j < samplesize_; j++) {
      GaussianNoiseArray_[i][j] = CLHEP::RandGaussQ::shoot(engine, NoiseMean, NoiseStd);
    }
  }
}

template <class DFr>
void HGCDigitizerBase<DFr>::run(std::unique_ptr<HGCDigitizerBase::DColl>& digiColl,
                                HGCSimHitDataAccumulator& simData,
                                const CaloSubdetectorGeometry* theGeom,
                                const std::unordered_set<DetId>& validIds,
                                uint32_t digitizationType,
                                CLHEP::HepRandomEngine* engine) {
  if (scaleByDose_)
    scal_.setGeometry(theGeom);
  if (NoiseGeneration_Method_ == true) {
    if (RandNoiseGenerationFlag_ == false) {
      GenerateGaussianNoise(engine, NoiseMean_, NoiseStd_);
      RandNoiseGenerationFlag_ = true;
    }
  }
  if (digitizationType == 0)
    runSimple(digiColl, simData, theGeom, validIds, engine);
  else
    runDigitizer(digiColl, simData, theGeom, validIds, digitizationType, engine);
}

template <class DFr>
void HGCDigitizerBase<DFr>::runSimple(std::unique_ptr<HGCDigitizerBase::DColl>& coll,
                                      HGCSimHitDataAccumulator& simData,
                                      const CaloSubdetectorGeometry* theGeom,
                                      const std::unordered_set<DetId>& validIds,
                                      CLHEP::HepRandomEngine* engine) {
  HGCSimHitData chargeColl, toa;

  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f);  //accumulated energy
  zeroData.hit_info[1].fill(0.f);  //time-of-flight
  std::array<double, samplesize_> cellNoiseArray;
  for (size_t i = 0; i < samplesize_; i++)
    cellNoiseArray[i] = 0.0;
  for (const auto& id : validIds) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = (simData.end() == it ? zeroData : it->second);
    addCellMetadata(cell, theGeom, id);
    if (NoiseGeneration_Method_ == true) {
      size_t hash_index = (CLHEP::RandFlat::shootInt(engine, (NoiseArrayLength_ - 1)) + id) % NoiseArrayLength_;

      cellNoiseArray = GaussianNoiseArray_[hash_index];
    }
    //set the noise,cce, LSB and threshold to be used
    float cce(1.f), noiseWidth(0.f), lsbADC(-1.f), maxADC(-1.f);
    // half the target mip value is the specification for ZS threshold
    uint32_t thrADC(std::floor(myFEelectronics_->getTargetMipValue() / 2));
    uint32_t gainIdx = 0;
    if (scaleByDose_) {
      HGCSiliconDetId detId(id);
      HGCalSiNoiseMap::SiCellOpCharacteristics siop =
          scal_.getSiCellOpCharacteristics(detId, HGCalSiNoiseMap::AUTO, myFEelectronics_->getTargetMipValue());
      cce = siop.cce;
      noiseWidth = siop.noise;
      lsbADC = scal_.getLSBPerGain()[(HGCalSiNoiseMap::GainRange_t)siop.gain];
      maxADC = scal_.getMaxADCPerGain()[(HGCalSiNoiseMap::GainRange_t)siop.gain];
      gainIdx = siop.gain;

      if (thresholdFollowsMIP_)
        thrADC = siop.thrADC;
    } else if (noise_fC_[cell.thickness - 1] != 0) {
      //this is kept for legacy compatibility with the TDR simulation
      //probably should simply be removed in a future iteration
      //note that in this legacy case, gainIdx is kept at 0, fixed
      cce = (cce_.empty() ? 1.f : cce_[cell.thickness - 1]);
      noiseWidth = cell.size * noise_fC_[cell.thickness - 1];
      thrADC =
          thresholdFollowsMIP_
              ? std::floor(cell.thickness * cce * myFEelectronics_->getADCThreshold() / myFEelectronics_->getADClsb())
              : std::floor(cell.thickness * myFEelectronics_->getADCThreshold() / myFEelectronics_->getADClsb());
    }

    //loop over time samples and add noise
    for (size_t i = 0; i < cell.hit_info[0].size(); i++) {
      double rawCharge(cell.hit_info[0][i]);

      //time of arrival
      toa[i] = cell.hit_info[1][i];
      if (myFEelectronics_->toaMode() == HGCFEElectronics<DFr>::WEIGHTEDBYE && rawCharge > 0)
        toa[i] = cell.hit_info[1][i] / rawCharge;

      //final charge estimation
      float noise;
      if (NoiseGeneration_Method_ == true)
        noise = (float)cellNoiseArray[i] * noiseWidth;
      else
        noise = CLHEP::RandGaussQ::shoot(engine, cellNoiseArray[i], noiseWidth);
      float totalCharge(rawCharge * cce + noise);
      if (totalCharge < 0.f)
        totalCharge = 0.f;
      chargeColl[i] = totalCharge;
    }

    //run the shaper to create a new data frame
    DFr rawDataFrame(id);
    int thickness = cell.thickness > 0 ? cell.thickness : 1;
    myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, engine, thrADC, lsbADC, gainIdx, maxADC, thickness);

    //update the output according to the final shape
    updateOutput(coll, rawDataFrame);
  }
}

template <class DFr>
void HGCDigitizerBase<DFr>::updateOutput(std::unique_ptr<HGCDigitizerBase::DColl>& coll, const DFr& rawDataFrame) {
  int itIdx(9);
  if (rawDataFrame.size() <= itIdx + 2)
    return;

  DFr dataFrame(rawDataFrame.id());
  dataFrame.resize(5);
  bool putInEvent(false);
  for (int it = 0; it < 5; it++) {
    dataFrame.setSample(it, rawDataFrame[itIdx - 2 + it]);
    if (it == 2)
      putInEvent = rawDataFrame[itIdx - 2 + it].threshold();
  }

  if (putInEvent) {
    coll->push_back(dataFrame);
  }
}

// cause the compiler to generate the appropriate code
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
template class HGCDigitizerBase<HGCEEDataFrame>;
template class HGCDigitizerBase<HGCBHDataFrame>;
template class HGCDigitizerBase<HGCalDataFrame>;
