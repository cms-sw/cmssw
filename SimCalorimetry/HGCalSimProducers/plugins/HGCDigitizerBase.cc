#include <memory>

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

using namespace hgc_digi;
using namespace hgc_digi_utils;

HGCDigitizerBase::HGCDigitizerBase(const edm::ParameterSet& ps)
    : scaleByDose_(false),
      det_(DetId::Forward),
      subdet_(ForwardSubdetector::ForwardEmpty),
      NoiseMean_(0.0),
      NoiseStd_(1.0) {
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
    scaleByDoseFactor_ = myCfg_.getParameter<edm::ParameterSet>("noise_fC").getParameter<double>("scaleByDoseFactor");
    doseMapFile_ = myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<std::string>("doseMap");
    scal_.setDoseMap(doseMapFile_, scaleByDoseAlgo);
    scal_.setFluenceScaleFactor(scaleByDoseFactor_);
    scalHFNose_.setDoseMap(doseMapFile_, scaleByDoseAlgo);
    scalHFNose_.setFluenceScaleFactor(scaleByDoseFactor_);
  } else {
    noise_fC_.resize(1, 1.f);
  }
  if (myCfg_.existsAs<edm::ParameterSet>("ileakParam")) {
    scal_.setIleakParam(
        myCfg_.getParameter<edm::ParameterSet>("ileakParam").template getParameter<std::vector<double>>("ileakParam"));
    scalHFNose_.setIleakParam(
        myCfg_.getParameter<edm::ParameterSet>("ileakParam").template getParameter<std::vector<double>>("ileakParam"));
  }
  if (myCfg_.existsAs<edm::ParameterSet>("cceParams")) {
    scal_.setCceParam(
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamFine"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThin"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThick"));
    scalHFNose_.setCceParam(
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamFine"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThin"),
        myCfg_.getParameter<edm::ParameterSet>("cceParams").template getParameter<std::vector<double>>("cceParamThick"));
  }

  edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
  myFEelectronics_ = std::make_unique<HGCFEElectronics<DFr>>(feCfg);
  myFEelectronics_->SetNoiseValues(noise_fC_);

  //override the "default ADC pulse" with the one with which was configured the FE electronics class
  scal_.setDefaultADCPulseShape(myFEelectronics_->getDefaultADCPulse());
  scalHFNose_.setDefaultADCPulseShape(myFEelectronics_->getDefaultADCPulse());

  RandNoiseGenerationFlag_ = false;
}

void HGCDigitizerBase::GenerateGaussianNoise(CLHEP::HepRandomEngine* engine,
                                             const double NoiseMean,
                                             const double NoiseStd) {
  for (size_t i = 0; i < NoiseArrayLength_; i++) {
    for (size_t j = 0; j < samplesize_; j++) {
      GaussianNoiseArray_[i][j] = CLHEP::RandGaussQ::shoot(engine, NoiseMean, NoiseStd);
    }
  }
}

void HGCDigitizerBase::run(std::unique_ptr<HGCDigitizerBase::DColl>& digiColl,
                           HGCSimHitDataAccumulator& simData,
                           const CaloSubdetectorGeometry* theGeom,
                           const std::unordered_set<DetId>& validIds,
                           uint32_t digitizationType,
                           CLHEP::HepRandomEngine* engine) {
  if (scaleByDose_) {
    scal_.setGeometry(theGeom, HGCalSiNoiseMap<HGCSiliconDetId>::AUTO, myFEelectronics_->getTargetMipValue());
    scalHFNose_.setGeometry(theGeom, HGCalSiNoiseMap<HFNoseDetId>::AUTO, myFEelectronics_->getTargetMipValue());
  }
  if (NoiseGeneration_Method_ == true) {
    if (RandNoiseGenerationFlag_ == false) {
      GenerateGaussianNoise(engine, NoiseMean_, NoiseStd_);
      RandNoiseGenerationFlag_ = true;
    }
  }
  myFEelectronics_->generateTimeOffset(engine);
  if (digitizationType == 0)
    runSimple(digiColl, simData, theGeom, validIds, engine);
  else
    runDigitizer(digiColl, simData, theGeom, validIds, engine);
}

void HGCDigitizerBase::runSimple(std::unique_ptr<HGCDigitizerBase::DColl>& coll,
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

    //set the noise,cce, LSB, threshold, and ADC pulse shape to be used
    float cce(1.f), noiseWidth(0.f), lsbADC(-1.f), maxADC(-1.f);
    // half the target mip value is the specification for ZS threshold
    uint32_t thrADC(std::floor(myFEelectronics_->getTargetMipValue() / 2));
    uint32_t gainIdx = 0;
    std::array<float, 6>& adcPulse = myFEelectronics_->getDefaultADCPulse();

    double tdcOnsetAuto = -1;
    if (scaleByDose_) {
      if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
        HGCalSiNoiseMap<HFNoseDetId>::SiCellOpCharacteristicsCore siop = scalHFNose_.getSiCellOpCharacteristicsCore(id);
        cce = siop.cce;
        noiseWidth = siop.noise;
        HGCalSiNoiseMap<HFNoseDetId>::GainRange_t gain((HGCalSiNoiseMap<HFNoseDetId>::GainRange_t)siop.gain);
        lsbADC = scalHFNose_.getLSBPerGain()[gain];
        maxADC = scalHFNose_.getMaxADCPerGain()[gain];
        adcPulse = scalHFNose_.adcPulseForGain(gain);
        gainIdx = siop.gain;
        tdcOnsetAuto = scal_.getTDCOnsetAuto(gainIdx);
        if (thresholdFollowsMIP_)
          thrADC = siop.thrADC;
      } else {
        HGCalSiNoiseMap<HGCSiliconDetId>::SiCellOpCharacteristicsCore siop = scal_.getSiCellOpCharacteristicsCore(id);
        cce = siop.cce;
        noiseWidth = siop.noise;
        HGCalSiNoiseMap<HGCSiliconDetId>::GainRange_t gain((HGCalSiNoiseMap<HGCSiliconDetId>::GainRange_t)siop.gain);
        lsbADC = scal_.getLSBPerGain()[gain];
        maxADC = scal_.getMaxADCPerGain()[gain];
        adcPulse = scal_.adcPulseForGain(gain);
        gainIdx = siop.gain;
        tdcOnsetAuto = scal_.getTDCOnsetAuto(gainIdx);
        if (thresholdFollowsMIP_)
          thrADC = siop.thrADC;
      }
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

    //loop over time samples, compute toa and add noise
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
    myFEelectronics_->runShaper(rawDataFrame,
                                chargeColl,
                                toa,
                                adcPulse,
                                engine,
                                thrADC,
                                lsbADC,
                                gainIdx,
                                maxADC,
                                thickness,
                                tdcOnsetAuto,
                                noiseWidth);

    //update the output according to the final shape
    updateOutput(coll, rawDataFrame);
  }
}

void HGCDigitizerBase::updateOutput(std::unique_ptr<HGCDigitizerBase::DColl>& coll, const DFr& rawDataFrame) {
  // 9th is the sample of hte intime amplitudes
  int itIdx(9);
  if (rawDataFrame.size() <= itIdx + 2)
    return;

  DFr dataFrame(rawDataFrame.id());
  dataFrame.resize(5);

  // if in time amplitude is above threshold
  // , then don't push back the dataframe
  if ((!rawDataFrame[itIdx].threshold())) {
    return;
  }

  for (int it = 0; it < 5; it++) {
    dataFrame.setSample(it, rawDataFrame[itIdx - 2 + it]);
  }

  coll->push_back(dataFrame);
}
