#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "CondFormats/HcalObjects/interface/HcalTPParameters.h"
#include "CondFormats/HcalObjects/interface/HcalTPChannelParameters.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

#include <iostream>

using namespace std;

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo(bool pf,
                                                   const std::vector<double>& w,
                                                   int latency,
                                                   uint32_t FG_threshold,
                                                   const std::vector<uint32_t>& FG_HF_thresholds,
                                                   uint32_t ZS_threshold,
                                                   int numberOfSamples,
                                                   int numberOfPresamples,
                                                   int numberOfFilterPresamplesHBQIE11,
                                                   int numberOfFilterPresamplesHEQIE11,
                                                   int numberOfSamplesHF,
                                                   int numberOfPresamplesHF,
                                                   bool useTDCInMinBiasBits,
                                                   uint32_t minSignalThreshold,
                                                   uint32_t PMT_NoiseThreshold)
    : incoder_(nullptr),
      outcoder_(nullptr),
      theThreshold(0),
      peakfind_(pf),
      weights_(w),
      latency_(latency),
      FG_threshold_(FG_threshold),
      FG_HF_thresholds_(FG_HF_thresholds),
      ZS_threshold_(ZS_threshold),
      numberOfSamples_(numberOfSamples),
      numberOfPresamples_(numberOfPresamples),
      numberOfFilterPresamplesHBQIE11_(numberOfFilterPresamplesHBQIE11),
      numberOfFilterPresamplesHEQIE11_(numberOfFilterPresamplesHEQIE11),
      numberOfSamplesHF_(numberOfSamplesHF),
      numberOfPresamplesHF_(numberOfPresamplesHF),
      useTDCInMinBiasBits_(useTDCInMinBiasBits),
      minSignalThreshold_(minSignalThreshold),
      PMT_NoiseThreshold_(PMT_NoiseThreshold),
      NCTScaleShift(0),
      RCTScaleShift(0),
      peak_finder_algorithm_(2),
      override_parameters_() {
  //No peak finding setting (for Fastsim)
  if (!peakfind_) {
    numberOfSamples_ = 1;
    numberOfPresamples_ = 0;
    numberOfSamplesHF_ = 1;
    numberOfPresamplesHF_ = 0;
  }
  // Switch to integer for comparisons - remove compiler warning
  ZS_threshold_I_ = ZS_threshold_;
}

HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo() {}

void HcalTriggerPrimitiveAlgo::setUpgradeFlags(bool hb, bool he, bool hf) {
  upgrade_hb_ = hb;
  upgrade_he_ = he;
  upgrade_hf_ = hf;
}

void HcalTriggerPrimitiveAlgo::setFixSaturationFlag(bool fix_saturation) { fix_saturation_ = fix_saturation; }

void HcalTriggerPrimitiveAlgo::overrideParameters(const edm::ParameterSet& ps) {
  override_parameters_ = ps;

  if (override_parameters_.exists("ADCThresholdHF")) {
    override_adc_hf_ = true;
    override_adc_hf_value_ = override_parameters_.getParameter<uint32_t>("ADCThresholdHF");
  }
  if (override_parameters_.exists("TDCMaskHF")) {
    override_tdc_hf_ = true;
    override_tdc_hf_value_ = override_parameters_.getParameter<unsigned long long>("TDCMaskHF");
  }
}

void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame& frame) {
  // TODO: Need to add support for seperate 28, 29 in HE
  //Hack for 300_pre10, should be removed.
  if (frame.id().depth() == 5)
    return;

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.size()));

  samples1.setPresamples(frame.presamples());
  incoder_->adc2Linear(frame, samples1);

  std::vector<bool> msb;
  incoder_->lookupMSB(frame, msb);

  if (ids.size() == 2) {
    // make a second trigprim for the other one, and split the energy
    IntegerCaloSamples samples2(ids[1], samples1.size());
    for (int i = 0; i < samples1.size(); ++i) {
      samples1[i] = uint32_t(samples1[i] * 0.5);
      samples2[i] = samples1[i];
    }
    samples2.setPresamples(frame.presamples());
    addSignal(samples2);
    addFG(ids[1], msb);
  }
  addSignal(samples1);
  addFG(ids[0], msb);
}

void HcalTriggerPrimitiveAlgo::addSignal(const HFDataFrame& frame) {
  if (frame.id().depth() == 1 || frame.id().depth() == 2) {
    std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(frame.id());
    std::vector<HcalTrigTowerDetId>::const_iterator it;
    for (it = ids.begin(); it != ids.end(); ++it) {
      HcalTrigTowerDetId trig_tower_id = *it;
      IntegerCaloSamples samples(trig_tower_id, frame.size());
      samples.setPresamples(frame.presamples());
      incoder_->adc2Linear(frame, samples);

      // Don't add to final collection yet
      // HF PMT veto sum is calculated in analyzerHF()
      IntegerCaloSamples zero_samples(trig_tower_id, frame.size());
      zero_samples.setPresamples(frame.presamples());
      addSignal(zero_samples);

      // Pre-LS1 Configuration
      if (trig_tower_id.version() == 0) {
        // Mask off depths: fgid is the same for both depths
        uint32_t fgid = (frame.id().maskDepth());

        if (theTowerMapFGSum.find(trig_tower_id) == theTowerMapFGSum.end()) {
          SumFGContainer sumFG;
          theTowerMapFGSum.insert(std::pair<HcalTrigTowerDetId, SumFGContainer>(trig_tower_id, sumFG));
        }

        SumFGContainer& sumFG = theTowerMapFGSum[trig_tower_id];
        SumFGContainer::iterator sumFGItr;
        for (sumFGItr = sumFG.begin(); sumFGItr != sumFG.end(); ++sumFGItr) {
          if (sumFGItr->id() == fgid) {
            break;
          }
        }
        // If find
        if (sumFGItr != sumFG.end()) {
          for (int i = 0; i < samples.size(); ++i) {
            (*sumFGItr)[i] += samples[i];
          }
        } else {
          //Copy samples (change to fgid)
          IntegerCaloSamples sumFGSamples(DetId(fgid), samples.size());
          sumFGSamples.setPresamples(samples.presamples());
          for (int i = 0; i < samples.size(); ++i) {
            sumFGSamples[i] = samples[i];
          }
          sumFG.push_back(sumFGSamples);
        }

        // set veto to true if Long or Short less than threshold
        if (HF_Veto.find(fgid) == HF_Veto.end()) {
          vector<bool> vetoBits(samples.size(), false);
          HF_Veto[fgid] = vetoBits;
        }
        for (int i = 0; i < samples.size(); ++i) {
          if (samples[i] < minSignalThreshold_) {
            HF_Veto[fgid][i] = true;
          }
        }
      }
      // HF 1x1
      else if (trig_tower_id.version() == 1) {
        uint32_t fgid = (frame.id().maskDepth());
        HFDetails& details = theHFDetailMap[trig_tower_id][fgid];
        // Check the frame type to determine long vs short
        if (frame.id().depth() == 1) {  // Long
          details.long_fiber = samples;
          details.LongDigi = frame;
        } else if (frame.id().depth() == 2) {  // Short
          details.short_fiber = samples;
          details.ShortDigi = frame;
        } else {
          // Neither long nor short... So we have no idea what to do
          edm::LogWarning("HcalTPAlgo") << "Unable to figure out what to do with data frame for " << frame.id();
          return;
        }
      }
      // Uh oh, we are in a bad/unknown state! Things will start crashing.
      else {
        return;
      }
    }
  }
}

void HcalTriggerPrimitiveAlgo::addSignal(const QIE10DataFrame& frame) {
  HcalDetId detId = frame.detid();
  // prevent QIE10 calibration channels from entering TP emulation
  if (detId.subdet() != HcalForward)
    return;

  auto ids = theTrigTowerGeometry->towerIds(frame.id());
  for (const auto& id : ids) {
    if (id.version() == 0) {
      edm::LogError("HcalTPAlgo") << "Encountered QIE10 data frame mapped to TP version 0:" << id;
      continue;
    }

    int nsamples = frame.samples();

    IntegerCaloSamples samples(id, nsamples);
    samples.setPresamples(frame.presamples());
    incoder_->adc2Linear(frame, samples);

    // Don't add to final collection yet
    // HF PMT veto sum is calculated in analyzerHF()
    IntegerCaloSamples zero_samples(id, nsamples);
    zero_samples.setPresamples(frame.presamples());
    addSignal(zero_samples);

    auto fid = HcalDetId(frame.id());
    auto& details = theHFUpgradeDetailMap[id][fid.maskDepth()];
    auto& detail = details[fid.depth() - 1];
    detail.samples = samples;
    detail.digi = frame;
    detail.validity.resize(nsamples);
    detail.passTDC.resize(nsamples);
    incoder_->lookupMSB(frame, detail.fgbits);
    for (int idx = 0; idx < nsamples; ++idx) {
      detail.validity[idx] = validChannel(frame, idx);
      detail.passTDC[idx] = passTDC(frame, idx);
    }
  }
}

void HcalTriggerPrimitiveAlgo::addSignal(const QIE11DataFrame& frame) {
  HcalDetId detId(frame.id());
  // prevent QIE11 calibration channels from entering TP emulation
  if (detId.subdet() != HcalEndcap && detId.subdet() != HcalBarrel)
    return;

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(detId);
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.samples()));

  samples1.setPresamples(frame.presamples());
  incoder_->adc2Linear(frame, samples1);

  std::vector<std::bitset<2>> msb(frame.samples(), 0);
  incoder_->lookupMSB(frame, msb);

  if (ids.size() == 2) {
    // make a second trigprim for the other one, and share the energy
    IntegerCaloSamples samples2(ids[1], samples1.size());
    for (int i = 0; i < samples1.size(); ++i) {
      samples1[i] = uint32_t(samples1[i]);
      samples2[i] = samples1[i];
    }
    samples2.setPresamples(frame.presamples());
    addSignal(samples2);
    addUpgradeFG(ids[1], detId.depth(), msb);
    addUpgradeTDCFG(ids[1], frame);
  }
  addSignal(samples1);
  addUpgradeFG(ids[0], detId.depth(), msb);
  addUpgradeTDCFG(ids[0], frame);
}

void HcalTriggerPrimitiveAlgo::addSignal(const IntegerCaloSamples& samples) {
  HcalTrigTowerDetId id(samples.id());
  SumMap::iterator itr = theSumMap.find(id);

  if (itr == theSumMap.end()) {
    theSumMap.insert(std::make_pair(id, samples));
  } else {
    // wish CaloSamples had a +=
    for (int i = 0; i < samples.size(); ++i) {
      (itr->second)[i] += samples[i];
    }
  }

  // if fix_saturation == true, keep track of tower with saturated input LUT
  if (fix_saturation_) {
    SatMap::iterator itr_sat = theSatMap.find(id);

    assert((itr == theSumMap.end()) == (itr_sat == theSatMap.end()));

    if (itr_sat == theSatMap.end()) {
      vector<bool> check_sat;
      for (int i = 0; i < samples.size(); ++i) {
        if (!(samples[i] < QIE11_LINEARIZATION_ET)) {
          check_sat.push_back(true);
        } else
          check_sat.push_back(false);
      }
      theSatMap.insert(std::make_pair(id, check_sat));
    } else {
      for (int i = 0; i < samples.size(); ++i) {
        if (!(samples[i] < QIE11_LINEARIZATION_ET))
          (itr_sat->second)[i] = true;
      }
    }
  }
}

void HcalTriggerPrimitiveAlgo::analyze(IntegerCaloSamples& samples, HcalTriggerPrimitiveDigi& result) {
  int shrink = weights_.size() - 1;
  std::vector<bool>& msb = fgMap_[samples.id()];
  IntegerCaloSamples sum(samples.id(), samples.size());

  //slide algo window
  for (int ibin = 0; ibin < int(samples.size()) - shrink; ++ibin) {
    int algosumvalue = 0;
    for (unsigned int i = 0; i < weights_.size(); i++) {
      //add up value * scale factor
      algosumvalue += int(samples[ibin + i] * weights_[i]);
    }
    if (algosumvalue < 0)
      sum[ibin] = 0;  // low-side
                      //high-side
    //else if (algosumvalue>QIE8_LINEARIZATION_ET) sum[ibin]=QIE8_LINEARIZATION_ET;
    else
      sum[ibin] = algosumvalue;  //assign value to sum[]
  }

  // Align digis and TP
  int dgPresamples = samples.presamples();
  int tpPresamples = numberOfPresamples_;
  int shift = dgPresamples - tpPresamples;
  int dgSamples = samples.size();
  int tpSamples = numberOfSamples_;
  if (peakfind_) {
    if ((shift < shrink) || (shift + tpSamples + shrink > dgSamples - (peak_finder_algorithm_ - 1))) {
      edm::LogInfo("HcalTriggerPrimitiveAlgo::analyze")
          << "TP presample or size from the configuration file is out of the accessible range. Using digi values from "
             "data instead...";
      shift = shrink;
      tpPresamples = dgPresamples - shrink;
      tpSamples = dgSamples - (peak_finder_algorithm_ - 1) - shrink - shift;
    }
  }

  std::vector<int> finegrain(tpSamples, false);

  IntegerCaloSamples output(samples.id(), tpSamples);
  output.setPresamples(tpPresamples);

  for (int ibin = 0; ibin < tpSamples; ++ibin) {
    // ibin - index for output TP
    // idx - index for samples + shift
    int idx = ibin + shift;

    //Peak finding
    if (peakfind_) {
      bool isPeak = false;
      switch (peak_finder_algorithm_) {
        case 1:
          isPeak = (samples[idx] > samples[idx - 1] && samples[idx] >= samples[idx + 1] && samples[idx] > theThreshold);
          break;
        case 2:
          isPeak = (sum[idx] > sum[idx - 1] && sum[idx] >= sum[idx + 1] && sum[idx] > theThreshold);
          break;
        default:
          break;
      }

      if (isPeak) {
        output[ibin] = std::min<unsigned int>(sum[idx], QIE8_LINEARIZATION_ET);
        finegrain[ibin] = msb[idx];
      }
      // Not a peak
      else
        output[ibin] = 0;
    } else {  // No peak finding, just output running sum
      output[ibin] = std::min<unsigned int>(sum[idx], QIE8_LINEARIZATION_ET);
      finegrain[ibin] = msb[idx];
    }

    // Only Pegged for 1-TS algo.
    if (peak_finder_algorithm_ == 1) {
      if (samples[idx] >= QIE8_LINEARIZATION_ET)
        output[ibin] = QIE8_LINEARIZATION_ET;
    }
  }
  outcoder_->compress(output, finegrain, result);
}

void HcalTriggerPrimitiveAlgo::analyzeQIE11(IntegerCaloSamples& samples,
                                            vector<bool> sample_saturation,
                                            HcalTriggerPrimitiveDigi& result,
                                            const HcalFinegrainBit& fg_algo) {
  HcalDetId detId(samples.id());

  // Get the |ieta| for current sample
  int theIeta = detId.ietaAbs();

  unsigned int dgSamples = samples.size();
  unsigned int dgPresamples = samples.presamples();

  unsigned int tpSamples = numberOfSamples_;
  unsigned int tpPresamples = numberOfPresamples_;

  unsigned int filterSamples = weightsQIE11_[theIeta].size();
  unsigned int filterPresamples = theIeta > theTrigTowerGeometry->topology().lastHBRing()
                                      ? numberOfFilterPresamplesHEQIE11_
                                      : numberOfFilterPresamplesHBQIE11_;

  unsigned int shift = dgPresamples - tpPresamples;

  // shrink keeps the FIR filter from going off the end of the 8TS vector
  unsigned int shrink = filterSamples - 1;

  auto& msb = fgUpgradeMap_[samples.id()];
  auto& timingTDC = fgUpgradeTDCMap_[samples.id()];
  IntegerCaloSamples sum(samples.id(), samples.size());

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(detId);

  // keep track of tower with saturated energy and force the total TP saturated
  bool force_saturation[samples.size()];
  for (int i = 0; i < samples.size(); i++) {
    force_saturation[i] = false;
  }

  //slide algo window
  for (unsigned int ibin = 0; ibin < dgSamples - shrink; ++ibin) {
    int algosumvalue = 0;
    bool check_sat = false;
    //TP energy calculation for PFA2
    if (weightsQIE11_[theIeta][0] == 255) {
      for (unsigned int i = 0; i < filterSamples; i++) {
        //add up value * scale factor
        // In addition, divide by two in the 10 degree phi segmentation region
        // to mimic 5 degree segmentation for the trigger
        unsigned int sample = samples[ibin + i];

        if (fix_saturation_ && (sample_saturation.size() > ibin + i))
          check_sat = (check_sat | sample_saturation[ibin + i] | (sample > QIE11_MAX_LINEARIZATION_ET));

        if (sample > QIE11_MAX_LINEARIZATION_ET)
          sample = QIE11_MAX_LINEARIZATION_ET;

        // Usually use a segmentation factor of 1.0 but for ieta >= 21 use 2
        int segmentationFactor = 1;
        if (ids.size() == 2) {
          segmentationFactor = 2;
        }

        algosumvalue += int(sample / segmentationFactor);
      }
      if (algosumvalue < 0)
        sum[ibin] = 0;  // low-side
                        //high-side
      //else if (algosumvalue>QIE11_LINEARIZATION_ET) sum[ibin]=QIE11_LINEARIZATION_ET;
      else
        sum[ibin] = algosumvalue;  //assign value to sum[]

      if (check_sat)
        force_saturation[ibin] = true;
      //TP energy calculation for PFA1' and PFA1
    } else {
      //add up value * scale factor
      // In addition, divide by two in the 10 degree phi segmentation region
      // to mimic 5 degree segmentation for the trigger
      int sampleTS = samples[ibin + 1];
      int sampleTSminus1 = samples[ibin];

      if (fix_saturation_ && (sample_saturation.size() > ibin + 1))
        check_sat = (sample_saturation[ibin + 1] || (sampleTS >= QIE11_MAX_LINEARIZATION_ET) ||
                     sample_saturation[ibin] || (sampleTSminus1 >= QIE11_MAX_LINEARIZATION_ET));

      if (sampleTS > QIE11_MAX_LINEARIZATION_ET)
        sampleTS = QIE11_MAX_LINEARIZATION_ET;

      if (sampleTSminus1 > QIE11_MAX_LINEARIZATION_ET)
        sampleTSminus1 = QIE11_MAX_LINEARIZATION_ET;

      // Usually use a segmentation factor of 1.0 but for ieta >= 21 use 2
      int segmentationFactor = 1;
      if (ids.size() == 2) {
        segmentationFactor = 2;
      }

      // Based on the |ieta| of the sample, retrieve the correct region weight
      int theWeight = weightsQIE11_[theIeta][0];

      algosumvalue = ((sampleTS << 8) - (sampleTSminus1 * theWeight)) / 256 / segmentationFactor;

      if (algosumvalue < 0)
        sum[ibin] = 0;  // low-side
                        //high-side
      //else if (algosumvalue>QIE11_LINEARIZATION_ET) sum[ibin]=QIE11_LINEARIZATION_ET;
      else
        sum[ibin] = algosumvalue;  //assign value to sum[]

      if (check_sat)
        force_saturation[ibin] = true;
    }
  }

  std::vector<int> finegrain(tpSamples, false);

  IntegerCaloSamples output(samples.id(), tpSamples);
  output.setPresamples(tpPresamples);

  for (unsigned int ibin = 0; ibin < tpSamples; ++ibin) {
    // ibin - index for output TP
    // idx - index for samples + shift - filterPresamples
    int idx = ibin + shift - filterPresamples;

    // When idx is <= 0 peakfind would compare out-of-bounds of the vector. Avoid this ambiguity
    if (idx <= 0) {
      output[ibin] = 0;
      continue;
    }

    //Only run the peak-finder when the PFA2 FIR filter is running, which corresponds to weights = 1
    if (weightsQIE11_[theIeta][0] == 255) {
      bool isPeak = (sum[idx] > sum[idx - 1] && sum[idx] >= sum[idx + 1] && sum[idx] > theThreshold);
      if (isPeak) {
        output[ibin] = std::min<unsigned int>(sum[idx], QIE11_MAX_LINEARIZATION_ET);

        if (fix_saturation_ && force_saturation[idx] && ids.size() == 2)
          output[ibin] = QIE11_MAX_LINEARIZATION_ET / 2;
        else if (fix_saturation_ && force_saturation[idx])
          output[ibin] = QIE11_MAX_LINEARIZATION_ET;

      } else {
        // Not a peak
        output[ibin] = 0;
      }
    } else {
      output[ibin] = std::min<unsigned int>(sum[idx], QIE11_MAX_LINEARIZATION_ET);

      if (fix_saturation_ && force_saturation[idx] && ids.size() == 2)
        output[ibin] = QIE11_MAX_LINEARIZATION_ET / 2;
      else if (fix_saturation_ && force_saturation[idx])
        output[ibin] = QIE11_MAX_LINEARIZATION_ET;
    }
    // peak-finding is not applied for FG bits
    // compute(msb) returns two bits (MIP). compute(timingTDC,ids) returns 6 bits (1 depth, 1 prompt, 1 delayed 01, 1 delayed 10, 2 reserved)
    finegrain[ibin] = fg_algo.compute(timingTDC[idx + filterPresamples], ids[0]).to_ulong() |
                      fg_algo.compute(msb[idx + filterPresamples]).to_ulong() << 4;
    if (ibin == tpPresamples && (idx + filterPresamples) != dgPresamples)
      edm::LogError("HcalTriggerPritimveAlgo")
          << "TP SOI (tpPresamples = " << tpPresamples
          << ") is not aligned with digi SOI (dgPresamples = " << dgPresamples << ")";
  }
  outcoder_->compress(output, finegrain, result);
}

void HcalTriggerPrimitiveAlgo::analyzeHF(IntegerCaloSamples& samples,
                                         HcalTriggerPrimitiveDigi& result,
                                         const int hf_lumi_shift) {
  HcalTrigTowerDetId detId(samples.id());

  // Align digis and TP
  int dgPresamples = samples.presamples();
  int tpPresamples = numberOfPresamplesHF_;
  int shift = dgPresamples - tpPresamples;
  int dgSamples = samples.size();
  int tpSamples = numberOfSamplesHF_;
  if (shift < 0 || shift + tpSamples > dgSamples) {
    edm::LogInfo("HcalTriggerPrimitiveAlgo::analyzeHF")
        << "TP presample or size from the configuration file is out of the accessible range. Using digi values from "
           "data instead...";
    tpPresamples = dgPresamples;
    shift = 0;
    tpSamples = dgSamples;
  }

  std::vector<int> finegrain(tpSamples, false);

  TowerMapFGSum::const_iterator tower2fg = theTowerMapFGSum.find(detId);
  assert(tower2fg != theTowerMapFGSum.end());

  const SumFGContainer& sumFG = tower2fg->second;
  // Loop over all L+S pairs that mapped from samples.id()
  // Note: 1 samples.id() = 6 x (L+S) without noZS
  for (SumFGContainer::const_iterator sumFGItr = sumFG.begin(); sumFGItr != sumFG.end(); ++sumFGItr) {
    const std::vector<bool>& veto = HF_Veto[sumFGItr->id().rawId()];
    for (int ibin = 0; ibin < tpSamples; ++ibin) {
      int idx = ibin + shift;
      // if not vetod, add L+S to total sum and calculate FG
      bool vetoed = idx < int(veto.size()) && veto[idx];
      if (!(vetoed && (*sumFGItr)[idx] > PMT_NoiseThreshold_)) {
        samples[idx] += (*sumFGItr)[idx];
        finegrain[ibin] = (finegrain[ibin] || (*sumFGItr)[idx] >= FG_threshold_);
      }
    }
  }

  IntegerCaloSamples output(samples.id(), tpSamples);
  output.setPresamples(tpPresamples);

  for (int ibin = 0; ibin < tpSamples; ++ibin) {
    int idx = ibin + shift;
    output[ibin] = samples[idx] >> hf_lumi_shift;
    static const int MAX_OUTPUT = QIE8_LINEARIZATION_ET;  // QIE8_LINEARIZATION_ET = 1023
    if (output[ibin] > MAX_OUTPUT)
      output[ibin] = MAX_OUTPUT;
  }
  outcoder_->compress(output, finegrain, result);
}

void HcalTriggerPrimitiveAlgo::analyzeHF2016(const IntegerCaloSamples& samples,
                                             HcalTriggerPrimitiveDigi& result,
                                             const int hf_lumi_shift,
                                             const HcalFeatureBit* embit) {
  // Align digis and TP
  const int SHIFT = samples.presamples() - numberOfPresamplesHF_;
  assert(SHIFT >= 0);
  assert((SHIFT + numberOfSamplesHF_) <= samples.size());

  // Try to find the HFDetails from the map corresponding to our samples
  const HcalTrigTowerDetId detId(samples.id());
  HFDetailMap::const_iterator it = theHFDetailMap.find(detId);
  // Missing values will give an empty digi
  if (it == theHFDetailMap.end()) {
    return;
  }

  std::vector<std::bitset<2>> finegrain(numberOfSamplesHF_, false);

  // Set up out output of IntergerCaloSamples
  IntegerCaloSamples output(samples.id(), numberOfSamplesHF_);
  output.setPresamples(numberOfPresamplesHF_);

  for (const auto& item : it->second) {
    auto& details = item.second;
    for (int ibin = 0; ibin < numberOfSamplesHF_; ++ibin) {
      const int IDX = ibin + SHIFT;
      int long_fiber_val = 0;
      if (IDX < details.long_fiber.size()) {
        long_fiber_val = details.long_fiber[IDX];
      }
      int short_fiber_val = 0;
      if (IDX < details.short_fiber.size()) {
        short_fiber_val = details.short_fiber[IDX];
      }
      output[ibin] += (long_fiber_val + short_fiber_val);

      uint32_t ADCLong = details.LongDigi[ibin].adc();
      uint32_t ADCShort = details.ShortDigi[ibin].adc();

      if (details.LongDigi.id().ietaAbs() >= FIRST_FINEGRAIN_TOWER) {
        finegrain[ibin][1] = (ADCLong > FG_HF_thresholds_[0] || ADCShort > FG_HF_thresholds_[0]);

        if (embit != nullptr)
          finegrain[ibin][0] = embit->fineGrainbit(details.ShortDigi, details.LongDigi, ibin);
      }
    }
  }

  for (int bin = 0; bin < numberOfSamplesHF_; ++bin) {
    static const unsigned int MAX_OUTPUT = QIE8_LINEARIZATION_ET;  // QIE8_LINEARIZATION_ET = 1023
    output[bin] = min({MAX_OUTPUT, output[bin] >> hf_lumi_shift});
  }

  std::vector<int> finegrain_converted;
  finegrain_converted.reserve(finegrain.size());
  for (const auto& fg : finegrain)
    finegrain_converted.push_back(fg.to_ulong());
  outcoder_->compress(output, finegrain_converted, result);
}

bool HcalTriggerPrimitiveAlgo::passTDC(const QIE10DataFrame& digi, int ts) const {
  auto parameters = conditions_->getHcalTPParameters();
  auto adc_threshold = parameters->getADCThresholdHF();
  auto tdc_mask = parameters->getTDCMaskHF();

  if (override_adc_hf_)
    adc_threshold = override_adc_hf_value_;
  if (override_tdc_hf_)
    tdc_mask = override_tdc_hf_value_;

  if (digi[ts].adc() < adc_threshold)
    return true;

  return (1ul << digi[ts].le_tdc()) & tdc_mask;
}

bool HcalTriggerPrimitiveAlgo::validChannel(const QIE10DataFrame& digi, int ts) const {
  // channels with invalid data should not contribute to the sum
  if (digi.linkError() || ts >= digi.samples() || !digi[ts].ok())
    return false;

  auto mask = conditions_->getHcalTPChannelParameter(HcalDetId(digi.id()))->getMask();
  if (mask)
    return false;

  return true;
}

void HcalTriggerPrimitiveAlgo::analyzeHFQIE10(const IntegerCaloSamples& samples,
                                              HcalTriggerPrimitiveDigi& result,
                                              const int hf_lumi_shift,
                                              const HcalFeatureBit* embit) {
  // Align digis and TP
  const int shift = samples.presamples() - numberOfPresamplesHF_;
  assert(shift >= 0);
  assert((shift + numberOfSamplesHF_) <= samples.size());
  assert(hf_lumi_shift >= 2);

  // Try to find the HFDetails from the map corresponding to our samples
  const HcalTrigTowerDetId detId(samples.id());
  auto it = theHFUpgradeDetailMap.find(detId);
  // Missing values will give an empty digi
  if (it == theHFUpgradeDetailMap.end()) {
    return;
  }

  std::vector<std::bitset<2>> finegrain(numberOfSamplesHF_, false);

  // Set up out output of IntergerCaloSamples
  IntegerCaloSamples output(samples.id(), numberOfSamplesHF_);
  output.setPresamples(numberOfPresamplesHF_);

  for (const auto& item : it->second) {
    auto& details = item.second;
    for (int ibin = 0; ibin < numberOfSamplesHF_; ++ibin) {
      const int idx = ibin + shift;

      int long_fiber_val = 0;
      int long_fiber_count = 0;
      int short_fiber_val = 0;
      int short_fiber_count = 0;

      bool saturated = false;

      for (auto i : {0, 2}) {
        if (idx < details[i].samples.size() and details[i].validity[idx] and details[i].passTDC[idx]) {
          long_fiber_val += details[i].samples[idx];
          saturated = saturated || (details[i].samples[idx] == QIE10_LINEARIZATION_ET);
          ++long_fiber_count;
        }
      }
      for (auto i : {1, 3}) {
        if (idx < details[i].samples.size() and details[i].validity[idx] and details[i].passTDC[idx]) {
          short_fiber_val += details[i].samples[idx];
          saturated = saturated || (details[i].samples[idx] == QIE10_LINEARIZATION_ET);
          ++short_fiber_count;
        }
      }

      if (saturated) {
        output[ibin] = QIE10_MAX_LINEARIZATION_ET;
      } else {
        // For details of the energy handling, see:
        // https://cms-docdb.cern.ch/cgi-bin/DocDB/ShowDocument?docid=12306
        // If both readouts are valid, average of the two energies is taken
        // division by 2 is compensated by adjusting the total scale shift in the end
        if (long_fiber_count == 2)
          long_fiber_val >>= 1;
        if (short_fiber_count == 2)
          short_fiber_val >>= 1;

        auto sum = long_fiber_val + short_fiber_val;
        // Similar to above, if both channels are valid,
        // average of the two energies is calculated
        // division by 2 here is also compensated by adjusting the total scale shift in the end
        if (long_fiber_count > 0 and short_fiber_count > 0)
          sum >>= 1;

        output[ibin] += sum;
      }

      for (const auto& detail : details) {
        if (idx < int(detail.digi.size()) and detail.validity[idx] and
            HcalDetId(detail.digi.id()).ietaAbs() >= FIRST_FINEGRAIN_TOWER) {
          if (useTDCInMinBiasBits_ && !detail.passTDC[idx])
            continue;
          finegrain[ibin][1] = finegrain[ibin][1] or detail.fgbits[idx][0];
          // what is commonly called the "second" HF min-bias bit is
          // actually the 0-th bit, which can also be used instead for the EM bit
          // (called finegrain[ibin][0] below) in non-HI running
          finegrain[ibin][0] = finegrain[ibin][0] or detail.fgbits[idx][1];
        }
      }
      // the EM bit is only used if the "second" FG bit is disabled
      if (embit != nullptr and FG_HF_thresholds_.at(1) != 255) {
        finegrain[ibin][0] = embit->fineGrainbit(details[1].digi,
                                                 details[3].digi,
                                                 details[0].digi,
                                                 details[2].digi,
                                                 details[1].validity[idx],
                                                 details[3].validity[idx],
                                                 details[0].validity[idx],
                                                 details[2].validity[idx],
                                                 idx);
      }
    }
  }

  for (int bin = 0; bin < numberOfSamplesHF_; ++bin) {
    output[bin] = min({(unsigned int)QIE10_MAX_LINEARIZATION_ET, output[bin] >> (hf_lumi_shift - 2)});
  }
  std::vector<int> finegrain_converted;
  finegrain_converted.reserve(finegrain.size());
  for (const auto& fg : finegrain)
    finegrain_converted.push_back(fg.to_ulong());
  outcoder_->compress(output, finegrain_converted, result);
}

void HcalTriggerPrimitiveAlgo::runZS(HcalTrigPrimDigiCollection& result) {
  for (HcalTrigPrimDigiCollection::iterator tp = result.begin(); tp != result.end(); ++tp) {
    bool ZS = true;
    for (int i = 0; i < tp->size(); ++i) {
      if (tp->sample(i).compressedEt() > ZS_threshold_I_) {
        ZS = false;
        break;
      }
    }
    if (ZS)
      tp->setZSInfo(false, true);
    else
      tp->setZSInfo(true, false);
  }
}

void HcalTriggerPrimitiveAlgo::runFEFormatError(const FEDRawDataCollection* rawraw,
                                                const HcalElectronicsMap* emap,
                                                HcalTrigPrimDigiCollection& result) {
  std::set<uint32_t> FrontEndErrors;

  for (int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; ++i) {
    const FEDRawData& raw = rawraw->FEDData(i);
    if (raw.size() < 12)
      continue;
    const HcalDCCHeader* dccHeader = (const HcalDCCHeader*)(raw.data());
    if (!dccHeader)
      continue;
    HcalHTRData htr;
    for (int spigot = 0; spigot < HcalDCCHeader::SPIGOT_COUNT; spigot++) {
      if (!dccHeader->getSpigotPresent(spigot))
        continue;
      dccHeader->getSpigotData(spigot, htr, raw.size());
      int dccid = dccHeader->getSourceId();
      int errWord = htr.getErrorsWord() & 0x1FFFF;
      bool HTRError = (!htr.check() || htr.isHistogramEvent() || (errWord & 0x800) != 0);

      if (HTRError) {
        bool valid = false;
        for (int fchan = 0; fchan < 3 && !valid; fchan++) {
          for (int fib = 0; fib < 9 && !valid; fib++) {
            HcalElectronicsId eid(fchan, fib, spigot, dccid - FEDNumbering::MINHCALFEDID);
            eid.setHTR(htr.readoutVMECrateId(), htr.htrSlot(), htr.htrTopBottom());
            DetId detId = emap->lookup(eid);
            if (detId.null())
              continue;
            HcalSubdetector subdet = (HcalSubdetector(detId.subdetId()));
            if (detId.det() != 4 || (subdet != HcalBarrel && subdet != HcalEndcap && subdet != HcalForward))
              continue;
            std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(detId);
            for (std::vector<HcalTrigTowerDetId>::const_iterator triggerId = ids.begin(); triggerId != ids.end();
                 ++triggerId) {
              FrontEndErrors.insert(triggerId->rawId());
            }
            //valid = true;
          }
        }
      }
    }
  }

  // Loop over TP collection
  // Set TP to zero if there is FE Format Error
  HcalTriggerPrimitiveSample zeroSample(0);
  for (HcalTrigPrimDigiCollection::iterator tp = result.begin(); tp != result.end(); ++tp) {
    if (FrontEndErrors.find(tp->id().rawId()) != FrontEndErrors.end()) {
      for (int i = 0; i < tp->size(); ++i)
        tp->setSample(i, zeroSample);
    }
  }
}

void HcalTriggerPrimitiveAlgo::addFG(const HcalTrigTowerDetId& id, std::vector<bool>& msb) {
  FGbitMap::iterator itr = fgMap_.find(id);
  if (itr != fgMap_.end()) {
    std::vector<bool>& _msb = itr->second;
    for (size_t i = 0; i < msb.size(); ++i)
      _msb[i] = _msb[i] || msb[i];
  } else
    fgMap_[id] = msb;
}

bool HcalTriggerPrimitiveAlgo::validUpgradeFG(const HcalTrigTowerDetId& id, int depth) const {
  if (depth > LAST_FINEGRAIN_DEPTH)
    return false;
  if (id.ietaAbs() > LAST_FINEGRAIN_TOWER)
    return false;
  if (id.ietaAbs() == HBHE_OVERLAP_TOWER and not upgrade_hb_)
    return false;
  return true;
}

bool HcalTriggerPrimitiveAlgo::needLegacyFG(const HcalTrigTowerDetId& id) const {
  // This tower (ietaAbs == 16) does not accept upgraded FG bits,
  // but needs pseudo legacy ones to ensure that the tower is processed
  // even when the QIE8 depths in front of it do not have energy deposits.
  if (id.ietaAbs() == HBHE_OVERLAP_TOWER and not upgrade_hb_)
    return true;
  return false;
}

bool HcalTriggerPrimitiveAlgo::needUpgradeID(const HcalTrigTowerDetId& id, int depth) const {
  // Depth 7 for TT 26, 27, and 28 is not considered a fine grain depth.
  // However, the trigger tower for these ieta should still be added to the fgUpgradeMap_
  // Otherwise, depth 7-only signal will not be analyzed.
  unsigned int aieta = id.ietaAbs();
  if (aieta >= FIRST_DEPTH7_TOWER and aieta <= LAST_FINEGRAIN_TOWER and depth > LAST_FINEGRAIN_DEPTH)
    return true;
  return false;
}

void HcalTriggerPrimitiveAlgo::addUpgradeFG(const HcalTrigTowerDetId& id,
                                            int depth,
                                            const std::vector<std::bitset<2>>& bits) {
  if (not validUpgradeFG(id, depth)) {
    if (needLegacyFG(id)) {
      std::vector<bool> pseudo(bits.size(), false);
      addFG(id, pseudo);
    } else if (needUpgradeID(id, depth)) {
      // If the tower id is not in the map yet
      // then for safety's sake add it, otherwise, no need
      // Likewise, we're here with non-fg depth 7 so the bits are not to be added
      auto it = fgUpgradeMap_.find(id);
      if (it == fgUpgradeMap_.end()) {
        FGUpgradeContainer element;
        element.resize(bits.size());
        fgUpgradeMap_.insert(std::make_pair(id, element));
      }
    }

    return;
  }

  auto it = fgUpgradeMap_.find(id);
  if (it == fgUpgradeMap_.end()) {
    FGUpgradeContainer element;
    element.resize(bits.size());
    it = fgUpgradeMap_.insert(std::make_pair(id, element)).first;
  }
  for (unsigned int i = 0; i < bits.size(); ++i) {
    it->second[i][0][depth - 1] = bits[i][0];
    it->second[i][1][depth - 1] = bits[i][1];
  }
}

void HcalTriggerPrimitiveAlgo::addUpgradeTDCFG(const HcalTrigTowerDetId& id, const QIE11DataFrame& frame) {
  HcalDetId detId(frame.id());
  if (detId.subdet() != HcalEndcap && detId.subdet() != HcalBarrel)
    return;

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(detId);
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.samples()));
  samples1.setPresamples(frame.presamples());
  incoder_->adc2Linear(frame, samples1);                                  // use linearization LUT
  std::vector<unsigned short> bits12_15 = incoder_->group0FGbits(frame);  // get 4 energy bits (12-15) from group 0 LUT

  bool is_compressed = false;
  if (detId.subdet() == HcalBarrel) {
    is_compressed = (frame.flavor() == 3);
    // 0 if frame.flavor is 0 (uncompressed), 1 if frame.flavor is 3 (compressed)
  }

  auto it = fgUpgradeTDCMap_.find(id);
  if (it == fgUpgradeTDCMap_.end()) {
    FGUpgradeTDCContainer element;
    element.resize(frame.samples());
    it = fgUpgradeTDCMap_.insert(std::make_pair(id, element)).first;
  }
  for (int i = 0; i < frame.samples(); i++) {
    it->second[i][detId.depth() - 1] =
        std::make_pair(std::make_pair(bits12_15[i], is_compressed), std::make_pair(frame[i].tdc(), samples1[i]));
  }
}

void HcalTriggerPrimitiveAlgo::setWeightsQIE11(const edm::ParameterSet& weightsQIE11) {
  // Names are just abs(ieta) for HBHE
  std::vector<std::string> ietaStrs = weightsQIE11.getParameterNames();
  for (auto& ietaStr : ietaStrs) {
    // Strip off "ieta" part of key and just use integer value in map
    auto const& v = weightsQIE11.getParameter<std::vector<int>>(ietaStr);
    weightsQIE11_[std::stoi(ietaStr.substr(4))] = {{v[0], v[1]}};
  }
}

void HcalTriggerPrimitiveAlgo::setWeightQIE11(int aieta, int weight) {
  // Simple map of |ieta| in HBHE to weight
  // Only one weight for SOI-1 TS
  weightsQIE11_[aieta] = {{weight, 255}};
}

void HcalTriggerPrimitiveAlgo::setPeakFinderAlgorithm(int algo) {
  if (algo <= 0 || algo > 2)
    throw cms::Exception("ERROR: Only algo 1 & 2 are supported.") << std::endl;
  peak_finder_algorithm_ = algo;
}

void HcalTriggerPrimitiveAlgo::setNCTScaleShift(int shift) { NCTScaleShift = shift; }

void HcalTriggerPrimitiveAlgo::setRCTScaleShift(int shift) { RCTScaleShift = shift; }
