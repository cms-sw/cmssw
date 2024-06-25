#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandGaussQ.h"

using namespace mtd;

ETLElectronicsSim::ETLElectronicsSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC)
    : geomToken_(iC.esConsumes()),
      geom_(nullptr),
      bxTime_(pset.getParameter<double>("bxTime")),
      integratedLum_(pset.getParameter<double>("IntegratedLuminosity")),
      adcNbits_(pset.getParameter<uint32_t>("adcNbits")),
      tdcNbits_(pset.getParameter<uint32_t>("tdcNbits")),
      adcSaturation_MIP_(pset.getParameter<double>("adcSaturation_MIP")),
      adcLSB_MIP_(adcSaturation_MIP_ / std::pow(2., adcNbits_)),
      adcBitSaturation_(std::pow(2, adcNbits_) - 1),
      adcThreshold_MIP_(pset.getParameter<double>("adcThreshold_MIP")),
      iThreshold_MIP_(pset.getParameter<double>("iThreshold_MIP")),
      toaLSB_ns_(pset.getParameter<double>("toaLSB_ns")),
      tdcBitSaturation_(std::pow(2, tdcNbits_) - 1),
      referenceChargeColl_(pset.getParameter<double>("referenceChargeColl")),
      noiseLevel_(pset.getParameter<double>("noiseLevel")),
      sigmaDistorsion_(pset.getParameter<double>("sigmaDistorsion")),
      sigmaTDC_(pset.getParameter<double>("sigmaTDC")),
      formulaLandauNoise_(pset.getParameter<std::string>("formulaLandauNoise")) {}

void ETLElectronicsSim::getEventSetup(const edm::EventSetup& evs) { geom_ = &evs.getData(geomToken_); }

void ETLElectronicsSim::run(const mtd::MTDSimHitDataAccumulator& input,
                            ETLDigiCollection& output,
                            CLHEP::HepRandomEngine* hre) const {
  MTDSimHitData chargeColl, toa1, toa2, tot;

  std::vector<double> emptyV;
  std::vector<double> radius(1);
  std::vector<double> fluence(1);
  std::vector<double> chOverMPV(1);

  for (MTDSimHitDataAccumulator::const_iterator it = input.begin(); it != input.end(); it++) {
    chargeColl.fill(0.f);
    toa1.fill(0.f);
    toa2.fill(0.f);
    tot.fill(0.f);

    ETLDetId detId = it->first.detid_;
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("EtlElectronicsSim") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                << detId.rawId() << ") is invalid!" << std::dec << std::endl;

    const PixelTopology& topo = static_cast<const PixelTopology&>(thedet->topology());
    Local3DPoint local_point(topo.localX(it->first.row_), topo.localY(it->first.column_), 0.);

    //Here we are sampling over all the buckets. However for the ETL there should be only one bucket at i = 0
    for (size_t i = 0; i < it->second.hit_info[0].size(); i++) {
      if ((it->second).hit_info[0][i] == 0) {
        continue;
      }
      if ((it->second).hit_info[0][i] < adcThreshold_MIP_)
        continue;
      // time of arrival
      double finalToA = (it->second).hit_info[1][i];
      double finalToC = (it->second).hit_info[1][i];

      // fill the time and charge arrays
      const unsigned int ibucket = std::floor(finalToA / bxTime_);
      if ((i + ibucket) >= chargeColl.size())
        continue;

      //Calculate the jitter
      double SignalToNoise =
          etlPulseShape_.maximum() * ((it->second).hit_info[0][i] / referenceChargeColl_) / noiseLevel_;
      double sigmaJitter1 = etlPulseShape_.timeOfMax() / SignalToNoise;
      double sigmaJitter2 = (etlPulseShape_.fallTime() - etlPulseShape_.timeOfMax()) / SignalToNoise;
      //Calculate the distorsion
      double sigmaDistorsion = sigmaDistorsion_;
      //Calculate the TDC
      double sigmaTDC = sigmaTDC_;
      //Calculate the Landau Noise
      chOverMPV[0] = (it->second).hit_info[0][i] / (it->second).hit_info[2][i];
      double sigmaLN = formulaLandauNoise_.evaluate(chOverMPV, emptyV);
      double sigmaToA = sqrt(sigmaJitter1 * sigmaJitter1 + sigmaDistorsion * sigmaDistorsion + sigmaTDC * sigmaTDC +
                             sigmaLN * sigmaLN);
      double sigmaToC = sqrt(sigmaJitter2 * sigmaJitter2 + sigmaDistorsion * sigmaDistorsion + sigmaTDC * sigmaTDC +
                             sigmaLN * sigmaLN);
      double smearing1 = 0.0;
      double smearing2 = 0.0;

      if (sigmaToA > 0. && sigmaToC > 0.) {
        smearing1 = CLHEP::RandGaussQ::shoot(hre, 0., sigmaToA);
        smearing2 = CLHEP::RandGaussQ::shoot(hre, 0., sigmaToC);
      }

      finalToA += smearing1;
      finalToC += smearing1 + smearing2;

      std::array<float, 3> times = etlPulseShape_.timeAtThr(
          (it->second).hit_info[0][i] / referenceChargeColl_, iThreshold_MIP_, iThreshold_MIP_);

      //The signal is below the threshold
      if (times[0] == 0 && times[1] == 0 && times[2] == 0) {
        continue;
      }
      //The signal is considered to be below the threshold
      finalToA += times[0];
      finalToC += times[2];
      if (finalToA >= finalToC)
        continue;
      chargeColl[i + ibucket] += (it->second).hit_info[0][i];

      if (toa1[i + ibucket] == 0. || (finalToA - ibucket * bxTime_) < toa1[i + ibucket]) {
        toa1[i + ibucket] = finalToA - ibucket * bxTime_;
        toa2[i + ibucket] = finalToC - ibucket * bxTime_;
      }

      tot[i + ibucket] = finalToC - finalToA;
    }
    // Run the shaper to create a new data frame
    ETLDataFrame rawDataFrame(it->first.detid_);
    runTrivialShaper(rawDataFrame, chargeColl, toa1, tot, it->first.row_, it->first.column_);
    updateOutput(output, rawDataFrame);
  }
}

void ETLElectronicsSim::runTrivialShaper(ETLDataFrame& dataFrame,
                                         const mtd::MTDSimHitData& chargeColl,
                                         const mtd::MTDSimHitData& toa,
                                         const mtd::MTDSimHitData& tot,
                                         const uint8_t row,
                                         const uint8_t col) const {
#ifdef EDM_ML_DEBUG
  bool dumpInfo(false);
  for (int it = 0; it < (int)(chargeColl.size()); it++) {
    if (chargeColl[it] > adcThreshold_MIP_) {
      dumpInfo = true;
      break;
    }
  }
  if (dumpInfo) {
    LogTrace("ETLElectronicsSim") << "[runTrivialShaper]";
  }
#endif

  //set new ADCs. Notice that we are only interested in the first element of the array for the ETL
  for (int it = 0; it < (int)(chargeColl.size()); it++) {
    //brute force saturation, maybe could to better with an exponential like saturation
    const uint32_t adc = std::min((uint32_t)std::floor(chargeColl[it] / adcLSB_MIP_), adcBitSaturation_);
    const uint32_t tdc_time1 = std::min((uint32_t)std::floor(toa[it] / toaLSB_ns_), tdcBitSaturation_);
    const uint32_t tdc_time2 = std::min((uint32_t)std::floor(tot[it] / toaLSB_ns_), tdcBitSaturation_);
    //If time over threshold is 0 the event is assumed to not pass the threshold
    bool thres = true;
    if (tdc_time2 == 0 || chargeColl[it] < adcThreshold_MIP_)
      thres = false;

    ETLSample newSample;
    newSample.set(thres, false, tdc_time1, tdc_time2, adc, row, col);

    //ETLSample newSample;
    dataFrame.setSample(it, newSample);

#ifdef EDM_ML_DEBUG
    if (dumpInfo) {
      LogTrace("ETLElectronicsSim") << adc << " (" << chargeColl[it] << "/" << adcLSB_MIP_ << ") ";
    }
#endif
  }

#ifdef EDM_ML_DEBUG
  if (dumpInfo) {
    std::ostringstream msg;
    dataFrame.print(msg);
    LogTrace("ETLElectronicsSim") << msg.str();
  }
#endif
}

void ETLElectronicsSim::updateOutput(ETLDigiCollection& coll, const ETLDataFrame& rawDataFrame) const {
  int itIdx(9);
  if (rawDataFrame.size() <= itIdx + 2)
    return;

  ETLDataFrame dataFrame(rawDataFrame.id());
  dataFrame.resize(dfSIZE);
  bool putInEvent(false);
  for (int it = 0; it < dfSIZE; ++it) {
    dataFrame.setSample(it, rawDataFrame[itIdx - 2 + it]);
    if (it == 2)
      putInEvent = rawDataFrame[itIdx - 2 + it].threshold();
  }

  if (putInEvent) {
    coll.push_back(dataFrame);
  }
}
