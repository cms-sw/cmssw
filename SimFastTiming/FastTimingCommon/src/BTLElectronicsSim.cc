#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

using namespace mtd;

BTLElectronicsSim::BTLElectronicsSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC)
    : bxTime_(pset.getParameter<double>("BunchCrossingTime")),
      lcepositionSlope_(pset.getParameter<double>("LCEpositionSlope")),
      sigmaLCEpositionSlope_(pset.getParameter<double>("SigmaLCEpositionSlope")),
      pulseT2Threshold_(pset.getParameter<double>("PulseT2Threshold")),
      pulseEThreshold_(pset.getParameter<double>("PulseEThrershold")),
      channelRearmMode_(pset.getParameter<uint32_t>("ChannelRearmMode")),
      channelRearmNClocks_(pset.getParameter<double>("ChannelRearmNClocks")),
      t1Delay_(pset.getParameter<double>("T1Delay")),
      sipmGain_(pset.getParameter<double>("SiPMGain")),
      paramPulseTbranchA_(pset.getParameter<std::vector<double>>("PulseTbranchAParam")),
      paramPulseEbranchA_(pset.getParameter<std::vector<double>>("PulseEbranchAParam")),
      paramThr1Rise_(pset.getParameter<std::vector<double>>("TimeAtThr1RiseParam")),
      paramThr2Rise_(pset.getParameter<std::vector<double>>("TimeAtThr2RiseParam")),
      paramTimeOverThr1_(pset.getParameter<std::vector<double>>("TimeOverThr1Param")),
      smearTimeForOOTtails_(pset.getParameter<bool>("SmearTimeForOOTtails")),
      scintillatorRiseTime_(pset.getParameter<double>("ScintillatorRiseTime")),
      scintillatorDecayTime_(pset.getParameter<double>("ScintillatorDecayTime")),
      stocasticParam_(pset.getParameter<std::vector<double>>("StocasticParam")),
      darkCountRate_(pset.getParameter<double>("DarkCountRate")),
      paramDCR_(pset.getParameter<std::vector<double>>("DCRParam")),
      sigmaElectronicNoise_(pset.getParameter<double>("SigmaElectronicNoise")),
      paramSR_(pset.getParameter<std::vector<double>>("SlewRateParam")),
      sigmaTDC_(pset.getParameter<double>("SigmaTDC")),
      sigmaClockGlobal_(pset.getParameter<double>("SigmaClockGlobal")),
      sigmaClockRU_(pset.getParameter<double>("SigmaClockRU")),
      paramPulseQ_(pset.getParameter<std::vector<double>>("PulseQParam")),
      paramPulseQRes_(pset.getParameter<std::vector<double>>("PulseQResParam")),
      corrCoeff_(pset.getParameter<double>("CorrelationCoefficient")),
      cosPhi_(0.5 * (sqrt(1. + corrCoeff_) + sqrt(1. - corrCoeff_))),
      sinPhi_(0.5 * corrCoeff_ / cosPhi_),
      scintillatorDecayTimeInv_(1. / scintillatorDecayTime_),
      sigmaConst2_(sigmaTDC_ * sigmaTDC_ + sigmaClockGlobal_ * sigmaClockGlobal_),
#ifdef EDM_ML_DEBUG
      debug_(true) {
#else
      debug_(false) {
#endif
#ifdef EDM_ML_DEBUG
  float lightOutput = 4.4f * pset.getParameter<double>("LightOutput");  // average Npe for 4.4 MeV
  float s1 = sigma_stochastic(lightOutput);
  float s2 = sigma_DCR(lightOutput);
  float s3 = sigma_electronics(lightOutput);
  float s4 = sigmaTDC_;
  float s5 = sqrt(sigmaClockGlobal_ * sigmaClockGlobal_ + sigmaClockRU_ * sigmaClockRU_);
  LogDebug("BTLElectronicsSim") << " BTL resolution model, for an average light output of " << std::fixed
                                << std::setw(14) << lightOutput << " :"
                                << "\n sigma stochastic   = " << std::setw(14) << s1
                                << "\n sigma DCR          = " << std::setw(14) << s2
                                << "\n sigma electronics  = " << std::setw(14) << s3
                                << "\n sigma digitization = " << std::setw(14) << s4
                                << "\n sigma clock        = " << std::setw(14) << s5 << "\n ---------------------"
                                << "\n sigma total        = " << std::setw(14)
                                << std::sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5);
#endif

  // --- Array to store a different local clock jitter for each readout unit
  smearingClockRU_ = new std::array<float, numberOfRUs_>();
  smearingClockRU_->fill(0.f);
}

BTLElectronicsSim::~BTLElectronicsSim() { delete smearingClockRU_; }

void BTLElectronicsSim::run(const mtd::MTDSimHitDataAccumulator& input,
                            BTLDigiCollection& output,
                            CLHEP::HepRandomEngine* hre) const {
  // --- Fill the readout-unit clock jitter array
  for (unsigned int iRU = 0; iRU < numberOfRUs_; ++iRU) {
    (*smearingClockRU_)[iRU] = CLHEP::RandGaussQ::shoot(hre, 0., sigmaClockRU_);
  }

  // --- Loop over the simhits (which have been propagated to the right and left sides of the crystal bar)
  for (MTDSimHitDataAccumulator::const_iterator it = input.begin(); it != input.end(); it++) {
    // --- Digitize only the in-time bucket
    const unsigned int iBX = mtd_digitizer::kInTimeBX;

    // --- Apply a common Npe Poisson fluctuation and independet Gaussian smearings
    //     for the LCE position slope to the right and left hits of the bar
    float npe[2] = {0.f, 0.f};

    // If both sides of the bar have an hit, the original simhit Npe and x can be determined:
    if ((it->second).hit_info[0][iBX] != 0. && (it->second).hit_info[2][iBX] != 0.) {
      float npe_origin = 0.5 * ((it->second).hit_info[0][iBX] + (it->second).hit_info[2][iBX]);
      float x_origin =
          0.5 * ((it->second).hit_info[0][iBX] - (it->second).hit_info[2][iBX]) / (npe_origin * lcepositionSlope_);

      float npe_fluctuated = CLHEP::RandPoissonQ::shoot(hre, npe_origin);

      float lceSlope_smearing = CLHEP::RandGaussQ::shoot(hre, 0., sigmaLCEpositionSlope_);
      npe[0] = npe_fluctuated * (1. + (sigmaLCEpositionSlope_ + lceSlope_smearing) * x_origin);

      lceSlope_smearing = CLHEP::RandGaussQ::shoot(hre, 0., sigmaLCEpositionSlope_);
      npe[1] = npe_fluctuated * (1. - (sigmaLCEpositionSlope_ + lceSlope_smearing) * x_origin);

    }
    // If there is a hit only on one side of the bar, the original simhit Npe and x can't be
    // determined and only a Poisson fluctuation to Npe_R or Npe_L is applied:
    else if ((it->second).hit_info[0][iBX] != 0. && (it->second).hit_info[2][iBX] == 0.) {
      npe[0] = CLHEP::RandPoissonQ::shoot(hre, (it->second).hit_info[0][iBX]);
    } else if ((it->second).hit_info[0][iBX] == 0. && (it->second).hit_info[2][iBX] != 0.) {
      npe[1] = CLHEP::RandPoissonQ::shoot(hre, (it->second).hit_info[2][iBX]);
    }
    // If there is no hit on either side of the bar, the hit is skipped:
    else {
      continue;
    }

    float charge_adc[2] = {0.f, 0.f};
    float toa1[2] = {0.f, 0.f};
    float toa2[2] = {0.f, 0.f};
    for (size_t iside = 0; iside < 2; iside++) {
      // --- Skip the empty buckets
      if (npe[iside] == 0.) {
        continue;
      }

      // ================================================================================
      //  TOFHiR's time branch
      // ================================================================================

      // --- Skip the hit if its amplitude is below the T2 threshold
      if (pulse_tbranch_uA(npe[iside]) < pulseT2Threshold_) {
        continue;
      }

      // --- Skip the hit if its amplitude is below the energy threshold
      if (pulse_ebranch_uA(npe[iside]) < pulseEThreshold_) {
        continue;
      }

      // --- Add the T1 and T2 threshold crossing times on the pulse rising edge to the SimHit time
      float finalToA1 = (it->second).hit_info[1 + 2 * iside][iBX] + time_at_Thr1Rise(npe[iside]);
      float finalToA2 = (it->second).hit_info[1 + 2 * iside][iBX] + time_at_Thr2Rise(npe[iside]);

      // --- Loop over the earlier OOT hits in the current bar to determine the channel
      //     rearming time and estimate the photon flux arriving at the in-time BX
      float channelRearmingTime = -bxTime_ * (mtd_digitizer::kInTimeBX - 1);
      float rate_oot = 0.;
      for (int ibx = 0; ibx < mtd_digitizer::kInTimeBX; ++ibx) {
        // Skip the OOT empty buckets
        if ((it->second).hit_info[2 * iside][ibx] == 0.) {
          continue;
        }

        float hit_time_oot = (it->second).hit_info[1 + 2 * iside][ibx];
        float hit_npe_oot = CLHEP::RandPoissonQ::shoot(hre, (it->second).hit_info[2 * iside][ibx]);

        // Calculate the channel rearming time for this hit (the hit is skipped if it
        // doesn't pass the T2 threshold or an earlier hit is holding the channel)
        float time_at_T1_oot = time_at_Thr1Rise(hit_npe_oot);

        if (channelRearmMode_ && pulse_tbranch_uA(hit_npe_oot) > pulseT2Threshold_ &&
            hit_time_oot + time_at_T1_oot > channelRearmingTime) {
          channelRearmingTime = rearming_time(hit_time_oot + time_at_T1_oot, hit_npe_oot);
        }

        // Rate of photons from earlier OOT hits in the current BTL cell
        if (smearTimeForOOTtails_) {
          rate_oot += hit_npe_oot * exp(hit_time_oot * scintillatorDecayTimeInv_) * scintillatorDecayTimeInv_;
        }

      }  // ibx loop

      // --- Skip the hit if the readout channel is not rearmed
      if (channelRearmMode_ && finalToA1 < channelRearmingTime) {
        continue;
      }

      // --- Uncertainty due to photons from earlier OOT hits in the current BTL cell
      if (smearTimeForOOTtails_ && rate_oot > 0.) {
        float sigma_oot = sqrt(rate_oot * scintillatorRiseTime_) * scintillatorDecayTime_ / npe[iside];
        float smearing_oot = CLHEP::RandGaussQ::shoot(hre, 0., sigma_oot);
        finalToA1 += smearing_oot;
        finalToA2 += smearing_oot;
      }

      // --- Stochastich term
      float sigmaStoc = sigma_stochastic(npe[iside]);
      finalToA1 += CLHEP::RandGaussQ::shoot(hre, 0., sigmaStoc);
      finalToA2 += CLHEP::RandGaussQ::shoot(hre, 0., sigmaStoc);

      // --- Add in quadrature the uncertainties due to the SiPM DCR and the electronic noise
      float sigmaDCR = sigma_DCR(npe[iside]);
      float sigmaElec = sigma_electronics(npe[iside]);
      float sigma2_tot_thr1 = sigmaDCR * sigmaDCR + sigmaElec * sigmaElec;

      // --- Add in quadrature the uncertainties independent of Npe: digitization and global clock distribution
      sigma2_tot_thr1 += sigmaConst2_;

      float sigma2_tot_thr2 = sigma2_tot_thr1;

      // --- Add the contribution due to the clock distribution within the readout units
      //     and smear the T1 and T2 arrival times assuming correlated uncertainties

      // Define a global readout-unit ID
      BTLDetId cellId((it->first).detid_);
      const int iRU = ((it->first).detid_ & BTLDetId::kBTLNewFormat
                           ? 12 * cellId.mtdRR() + 6 * cellId.mtdSide() + cellId.runit()
                           : 12 * (cellId.mtdRR() - 1) + 6 * cellId.mtdSide() + cellId.runit() - 1);

      float smearing_thr1_uncorr = CLHEP::RandGaussQ::shoot(hre, 0., sqrt(sigma2_tot_thr1)) + (*smearingClockRU_)[iRU];
      float smearing_thr2_uncorr = CLHEP::RandGaussQ::shoot(hre, 0., sqrt(sigma2_tot_thr2)) + (*smearingClockRU_)[iRU];

      finalToA1 += cosPhi_ * smearing_thr1_uncorr + sinPhi_ * smearing_thr2_uncorr;
      finalToA2 += sinPhi_ * smearing_thr1_uncorr + cosPhi_ * smearing_thr2_uncorr;

      toa1[iside] = finalToA1;
      toa2[iside] = finalToA2;

      // ================================================================================
      //  TOFHiR's energy branch
      // ================================================================================

      // --- Get the pulse amplitude in ADC counts
      float amp = pulse_q(npe[iside]);

      // --- Get the average uncertainty on the pulse amplitude (here the unsmeared
      //     value of Npe is used, because the parameterization of the relative
      //     amplitude resolution already includes the photostatistics fluctuation)
      float sigma_amp = amp * pulse_qRes((it->second).hit_info[2 * iside][iBX]);

      charge_adc[iside] = CLHEP::RandGaussQ::shoot(hre, amp, sigma_amp);

    }  // iside loop

    // --- skip if both sides are empty
    if (charge_adc[0] == 0 && charge_adc[1] == 0)
      continue;
    if (toa1[0] == 0 && toa1[1] == 0)
      continue;

    // --- Run the shaper to create a new data frame
    BTLDataFrame rawDataFrame(it->first.detid_);
    runTrivialShaper(rawDataFrame, charge_adc, toa1, toa2, it->first.row_, it->first.column_);
    updateOutput(output, rawDataFrame);

  }  // MTDSimHitDataAccumulator loop
}

void BTLElectronicsSim::runTrivialShaper(BTLDataFrame& dataFrame,
                                         const float (&charge_adc)[2],
                                         const float (&toa1)[2],
                                         const float (&toa2)[2],
                                         const uint8_t row,
                                         const uint8_t col) const {
  bool debug = debug_;

  if (debug) {
    LogTrace("BTLElectronicsSim") << "[runTrivialShaper] DetId " << dataFrame.id().rawId() << std::endl;
  }

  // --- Digitize the hit charge and times
  for (int iside = 0; iside < dfSIZE; iside++) {
    BTLSample newSample;
    newSample.set(false, false, 0, 0, 0, row, col);

    //brute force saturation, maybe could to better with an exponential like saturation
    const uint32_t adc = std::min((uint32_t)std::round(charge_adc[iside]), adcBitSaturation_);
    const uint32_t tdc_time1 = std::min((uint32_t)std::round(toa1[iside] / tdcLSB_ns_), tdcBitSaturation_);
    const uint32_t tdc_time2 = std::min((uint32_t)std::round(toa2[iside] / tdcLSB_ns_), tdcBitSaturation_);

    newSample.set(true, tdc_time1 == tdcBitSaturation_, tdc_time2, tdc_time1, adc, row, col);
    dataFrame.setSample(iside, newSample);

    if (debug) {
      LogTrace("BTLElectronicsSim") << "Side " << iside << ": ADC = " << adc << " (" << charge_adc[iside] << "), "
                                    << "TDC1 = " << tdc_time1 << " (" << toa1[iside] << "), "
                                    << "TDC2 = " << tdc_time2 << " (" << toa2[iside] << ")" << std::endl;
    }
  }  // iside loop

  if (debug) {
    std::ostringstream msg;
    dataFrame.print(msg);
    LogTrace("BTLElectronicsSim") << msg.str() << std::endl;
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

float BTLElectronicsSim::rearming_time(const float& hit_time, const float& hit_npe) const {
  // mode 1: the channel is rearmed after the falling edge of the trigger_B signal
  // mode 2: the channel is rearmed after n cycles of the TOFHiR clock
  float deadTime = t1Delay_ + (channelRearmMode_ == 1 ? time_over_Thr1(hit_npe) : channelRearmNClocks_ * tofhirClock_);

  // Sync the rearming time with the next rising edge of the TOFHiR clock
  return (std::round((hit_time + deadTime) / tofhirClock_) + 1.) * tofhirClock_;
}

float BTLElectronicsSim::pulse_tbranch_uA(const float& npe) const {
  float gainXnpe = sipmGain_ * npe;
  return paramPulseTbranchA_[0] + paramPulseTbranchA_[1] * gainXnpe;
}

float BTLElectronicsSim::pulse_ebranch_uA(const float& npe) const {
  float gainXnpe = sipmGain_ * npe;
  return paramPulseEbranchA_[0] + paramPulseEbranchA_[1] * gainXnpe;
}

float BTLElectronicsSim::time_at_Thr1Rise(const float& npe) const {
  return paramThr1Rise_[0] * std::pow(sipmGain_ * npe, paramThr1Rise_[1]);
}

float BTLElectronicsSim::time_at_Thr2Rise(const float& npe) const {
  return paramThr2Rise_[0] * std::pow(sipmGain_ * npe, paramThr2Rise_[1]);
}

float BTLElectronicsSim::time_over_Thr1(const float& npe) const {
  float gainXnpe = sipmGain_ * npe;

  float time_over_thr1 =
      (gainXnpe <= paramTimeOverThr1_[0]
           ? paramTimeOverThr1_[4] * gainXnpe * gainXnpe * gainXnpe + paramTimeOverThr1_[3] * gainXnpe * gainXnpe +
                 paramTimeOverThr1_[2] * gainXnpe + paramTimeOverThr1_[1]
           : paramTimeOverThr1_[5] * gainXnpe + paramTimeOverThr1_[6]);

  return time_over_thr1;
}

float BTLElectronicsSim::sigma_stochastic(const float& npe) const {
  // Trick to safely switch off the stochastic contribution for resolution studies:
  if (stocasticParam_[0] == 0.) {
    return 0.;
  }

  return sqrt2_ * stocasticParam_[0] *
         std::pow(npe, -stocasticParam_[1]);  // The uncertainty is provided for the combination of two SiPMs
}

float BTLElectronicsSim::sigma_DCR(const float& npe) const {
  // Trick to safely switch off the electronics contribution for resolution studies:
  if (darkCountRate_ == 0.) {
    return 0.;
  }

  return sqrt2_ * paramDCR_[0] * std::pow(darkCountRate_, paramDCR_[1]) /
         npe;  // The uncertainty is provided for the combination of two SiPMs
}

float BTLElectronicsSim::sigma_electronics(const float& npe) const {
  // Trick to safely switch off the electronics contribution for resolution studies:
  if (sigmaElectronicNoise_ == 0.) {
    return 0.;
  }

  float gainXnpe = sipmGain_ * npe;
  float res = sigmaElectronicNoise_;

  if (gainXnpe <= paramSR_[0]) {
    res /= (paramSR_[2] * gainXnpe + paramSR_[1]);
  } else {
    res /= (paramSR_[3] * std::log(gainXnpe) + paramSR_[2] * paramSR_[0] - paramSR_[3] * std::log(paramSR_[0]) +
            paramSR_[1]);
  }

  return std::sqrt(res * res);
}

float BTLElectronicsSim::pulse_q(const float& npe) const { return paramPulseQ_[0] + paramPulseQ_[1] * npe; }

float BTLElectronicsSim::pulse_qRes(const float& npe) const {
  return paramPulseQRes_[0] * std::pow(npe, paramPulseQRes_[1]);
}
