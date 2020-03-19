#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

using namespace mtd;

BTLElectronicsSim::BTLElectronicsSim(const edm::ParameterSet& pset)
    : debug_(pset.getUntrackedParameter<bool>("debug", false)),
      bxTime_(pset.getParameter<double>("bxTime")),
      testBeamMIPTimeRes_(pset.getParameter<double>("TestBeamMIPTimeRes")),
      ScintillatorRiseTime_(pset.getParameter<double>("ScintillatorRiseTime")),
      ScintillatorDecayTime_(pset.getParameter<double>("ScintillatorDecayTime")),
      ChannelTimeOffset_(pset.getParameter<double>("ChannelTimeOffset")),
      smearChannelTimeOffset_(pset.getParameter<double>("smearChannelTimeOffset")),
      EnergyThreshold_(pset.getParameter<double>("EnergyThreshold")),
      TimeThreshold1_(pset.getParameter<double>("TimeThreshold1")),
      TimeThreshold2_(pset.getParameter<double>("TimeThreshold2")),
      ReferencePulseNpe_(pset.getParameter<double>("ReferencePulseNpe")),
      SinglePhotonTimeResolution_(pset.getParameter<double>("SinglePhotonTimeResolution")),
      DarkCountRate_(pset.getParameter<double>("DarkCountRate")),
      SigmaElectronicNoise_(pset.getParameter<double>("SigmaElectronicNoise")),
      SigmaClock_(pset.getParameter<double>("SigmaClock")),
      smearTimeForOOTtails_(pset.getParameter<bool>("SmearTimeForOOTtails")),
      Npe_to_pC_(pset.getParameter<double>("Npe_to_pC")),
      Npe_to_V_(pset.getParameter<double>("Npe_to_V")),
      adcNbits_(pset.getParameter<uint32_t>("adcNbits")),
      tdcNbits_(pset.getParameter<uint32_t>("tdcNbits")),
      adcSaturation_MIP_(pset.getParameter<double>("adcSaturation_MIP")),
      adcBitSaturation_(std::pow(2, adcNbits_) - 1),
      adcLSB_MIP_(adcSaturation_MIP_ / adcBitSaturation_),
      adcThreshold_MIP_(pset.getParameter<double>("adcThreshold_MIP")),
      toaLSB_ns_(pset.getParameter<double>("toaLSB_ns")),
      tdcBitSaturation_(std::pow(2, tdcNbits_) - 1),
      CorrCoeff_(pset.getParameter<double>("CorrelationCoefficient")),
      cosPhi_(0.5 * (sqrt(1. + CorrCoeff_) + sqrt(1. - CorrCoeff_))),
      sinPhi_(0.5 * CorrCoeff_ / cosPhi_),
      ScintillatorDecayTime2_(ScintillatorDecayTime_ * ScintillatorDecayTime_),
      ScintillatorDecayTimeInv_(1. / ScintillatorDecayTime_),
      SPTR2_(SinglePhotonTimeResolution_ * SinglePhotonTimeResolution_),
      DCRxRiseTime_(DarkCountRate_ * ScintillatorRiseTime_),
      SigmaElectronicNoise2_(SigmaElectronicNoise_ * SigmaElectronicNoise_),
      SigmaClock2_(SigmaClock_ * SigmaClock_) {}

void BTLElectronicsSim::run(const mtd::MTDSimHitDataAccumulator& input,
                            BTLDigiCollection& output,
                            CLHEP::HepRandomEngine* hre) const {
  MTDSimHitData chargeColl, toa1, toa2;

  for (MTDSimHitDataAccumulator::const_iterator it = input.begin(); it != input.end(); it++) {
    // --- Digitize only the in-time bucket:
    const unsigned int iBX = mtd_digitizer::kInTimeBX;

    chargeColl.fill(0.f);
    toa1.fill(0.f);
    toa2.fill(0.f);
    for (size_t iside = 0; iside < 2; iside++) {
      // --- Fluctuate the total number of photo-electrons
      float Npe = CLHEP::RandPoissonQ::shoot(hre, (it->second).hit_info[2 * iside][iBX]);
      if (Npe < EnergyThreshold_)
        continue;

      // --- Get the time of arrival and add a channel time offset
      float finalToA1 = (it->second).hit_info[1 + 2 * iside][iBX] + ChannelTimeOffset_;

      if (smearChannelTimeOffset_ > 0.) {
        float timeSmearing = CLHEP::RandGaussQ::shoot(hre, 0., smearChannelTimeOffset_);
        finalToA1 += timeSmearing;
      }

      // --- Calculate and add the time walk: the time of arrival is read in correspondence
      //                                      with two thresholds on the signal pulse
      std::array<float, 3> times =
          btlPulseShape_.timeAtThr(Npe / ReferencePulseNpe_, TimeThreshold1_ * Npe_to_V_, TimeThreshold2_ * Npe_to_V_);

      // --- If the pulse amplitude is smaller than TimeThreshold2, the trigger does not fire
      if (times[1] == 0.)
        continue;

      float finalToA2 = finalToA1 + times[1];
      finalToA1 += times[0];

      // --- Estimate the time uncertainty due to photons from earlier OOT hits in the current BTL cell
      if (smearTimeForOOTtails_) {
        float rate_oot = 0.;
        // Loop on earlier OOT hits
        for (int ibx = 0; ibx < mtd_digitizer::kInTimeBX; ++ibx) {
          if ((it->second).hit_info[2 * iside][ibx] > 0.) {
            float hit_time = (it->second).hit_info[1 + 2 * iside][ibx] + bxTime_ * (ibx - mtd_digitizer::kInTimeBX);
            float npe_oot = CLHEP::RandPoissonQ::shoot(hre, (it->second).hit_info[2 * iside][ibx]);
            rate_oot += npe_oot * exp(hit_time * ScintillatorDecayTimeInv_) * ScintillatorDecayTimeInv_;
          }
        }  // ibx loop

        if (rate_oot > 0.) {
          float sigma_oot = sqrt(rate_oot * ScintillatorRiseTime_) * ScintillatorDecayTime_ / Npe;
          float smearing_oot = CLHEP::RandGaussQ::shoot(hre, 0., sigma_oot);
          finalToA1 += smearing_oot;
          finalToA2 += smearing_oot;
        }
      }  // if smearTimeForOOTtails_

      // --- Uncertainty due to the fluctuations of the n-th photon arrival time:
      if (testBeamMIPTimeRes_ > 0.) {
        // In this case the time resolution is parametrized from the testbeam.
        // The same parameterization is used for both thresholds.
        float sigma = testBeamMIPTimeRes_ / sqrt(Npe);
        float smearing_stat_thr1 = CLHEP::RandGaussQ::shoot(hre, 0., sigma);
        float smearing_stat_thr2 = CLHEP::RandGaussQ::shoot(hre, 0., sigma);

        finalToA1 += smearing_stat_thr1;
        finalToA2 += smearing_stat_thr2;

      } else {
        // In this case the time resolution is taken from the literature.
        // The fluctuations due to the first TimeThreshold1_ p.e. are common to both times
        float smearing_stat_thr1 =
            CLHEP::RandGaussQ::shoot(hre, 0., ScintillatorDecayTime_ * sqrt(sigma2_pe(TimeThreshold1_, Npe)));
        float smearing_stat_thr2 = CLHEP::RandGaussQ::shoot(
            hre, 0., ScintillatorDecayTime_ * sqrt(sigma2_pe(TimeThreshold2_ - TimeThreshold1_, Npe)));
        finalToA1 += smearing_stat_thr1;
        finalToA2 += smearing_stat_thr1 + smearing_stat_thr2;
      }

      // --- Add in quadrature the uncertainties due to the SiPM timing resolution, the SiPM DCR,
      //     the electronic noise and the clock distribution:
      float slew2 = ScintillatorDecayTime2_ / Npe / Npe;

      float sigma2_tot_thr1 =
          SPTR2_ / TimeThreshold1_ + (DCRxRiseTime_ + SigmaElectronicNoise2_) * slew2 + SigmaClock2_;
      float sigma2_tot_thr2 =
          SPTR2_ / TimeThreshold2_ + (DCRxRiseTime_ + SigmaElectronicNoise2_) * slew2 + SigmaClock2_;

      // --- Smear the arrival times using the correlated uncertainties:
      float smearing_thr1_uncorr = CLHEP::RandGaussQ::shoot(hre, 0., sqrt(sigma2_tot_thr1));
      float smearing_thr2_uncorr = CLHEP::RandGaussQ::shoot(hre, 0., sqrt(sigma2_tot_thr2));

      finalToA1 += cosPhi_ * smearing_thr1_uncorr + sinPhi_ * smearing_thr2_uncorr;
      finalToA2 += sinPhi_ * smearing_thr1_uncorr + cosPhi_ * smearing_thr2_uncorr;

      chargeColl[iside] = Npe * Npe_to_pC_;  // the p.e. number is here converted to pC

      toa1[iside] = finalToA1;
      toa2[iside] = finalToA2;

    }  // iside loop

    //run the shaper to create a new data frame
    BTLDataFrame rawDataFrame(it->first.detid_);
    runTrivialShaper(rawDataFrame, chargeColl, toa1, toa2, it->first.row_, it->first.column_);
    updateOutput(output, rawDataFrame);

  }  // MTDSimHitDataAccumulator loop
}

void BTLElectronicsSim::runTrivialShaper(BTLDataFrame& dataFrame,
                                         const mtd::MTDSimHitData& chargeColl,
                                         const mtd::MTDSimHitData& toa1,
                                         const mtd::MTDSimHitData& toa2,
                                         const uint8_t row,
                                         const uint8_t col) const {
  bool debug = debug_;
#ifdef EDM_ML_DEBUG
  for (int it = 0; it < (int)(chargeColl.size()); it++)
    debug |= (chargeColl[it] > adcThreshold_fC_);
#endif

  if (debug)
    edm::LogVerbatim("BTLElectronicsSim") << "[runTrivialShaper]" << std::endl;

  //set new ADCs
  for (int it = 0; it < (int)(chargeColl.size()); it++) {
    BTLSample newSample;
    newSample.set(false, false, 0, 0, 0, row, col);

    //brute force saturation, maybe could to better with an exponential like saturation
    const uint32_t adc = std::min((uint32_t)std::floor(chargeColl[it] / adcLSB_MIP_), adcBitSaturation_);
    const uint32_t tdc_time1 = std::min((uint32_t)std::floor(toa1[it] / toaLSB_ns_), tdcBitSaturation_);
    const uint32_t tdc_time2 = std::min((uint32_t)std::floor(toa2[it] / toaLSB_ns_), tdcBitSaturation_);

    newSample.set(
        chargeColl[it] > adcThreshold_MIP_, tdc_time1 == tdcBitSaturation_, tdc_time2, tdc_time1, adc, row, col);
    dataFrame.setSample(it, newSample);

    if (debug)
      edm::LogVerbatim("BTLElectronicsSim") << adc << " (" << chargeColl[it] << "/" << adcLSB_MIP_ << ") ";
  }

  if (debug) {
    std::ostringstream msg;
    dataFrame.print(msg);
    edm::LogVerbatim("BTLElectronicsSim") << msg.str() << std::endl;
  }
}

void BTLElectronicsSim::updateOutput(BTLDigiCollection& coll, const BTLDataFrame& rawDataFrame) const {
  BTLDataFrame dataFrame(rawDataFrame.id());
  dataFrame.resize(dfSIZE);
  bool putInEvent(false);
  for (int it = 0; it < dfSIZE; ++it) {
    dataFrame.setSample(it, rawDataFrame[it]);
    if (it == 0)
      putInEvent = rawDataFrame[it].threshold();
  }

  if (putInEvent) {
    coll.push_back(dataFrame);
  }
}

float BTLElectronicsSim::sigma2_pe(const float& Q, const float& R) const {
  float OneOverR = 1. / R;
  float OneOverR2 = OneOverR * OneOverR;

  // --- This is Eq. (17) from Nucl. Instr. Meth. A 564 (2006) 185
  float sigma2 = Q * OneOverR2 *
                 (1. + 2. * (Q + 1.) * OneOverR + (Q + 1.) * (6. * Q + 11) * OneOverR2 +
                  (Q + 1.) * (Q + 2.) * (2. * Q + 5.) * OneOverR2 * OneOverR);

  return sigma2;
}
