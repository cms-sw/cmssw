// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.
// Modified 15/May/2013 mark.grimes@bristol.ac.uk - Modified so that the digi-sim link has the correct
// index for the sim hits stored. It was previously always set to zero (I won't mention that it was
// me who originally wrote that).
// Modified on Feb 11, 2015: prolay.kumar.mal@cern.ch & Jean-Laurent.Agram@cern.ch
//                           Added/Modified the individual strip noise in zero suppression
//                           mode from the conditions DB; previously, the digitizer used to
//                           consider the noise value for individual strips inside a module from
//                           the central strip noise value.
//////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SiStripDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CLHEP/Random/RandFlat.h"

#include <cstring>
#include <sstream>

#include <boost/algorithm/string.hpp>

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : theThreshold(conf.getParameter<double>("NoiseSigmaThreshold")),
      cmnRMStib(conf.getParameter<double>("cmnRMStib")),
      cmnRMStob(conf.getParameter<double>("cmnRMStob")),
      cmnRMStid(conf.getParameter<double>("cmnRMStid")),
      cmnRMStec(conf.getParameter<double>("cmnRMStec")),
      APVSaturationProbScaling_(conf.getParameter<double>("APVSaturationProbScaling")),
      makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", false)),
      peakMode(conf.getParameter<bool>("APVpeakmode")),
      noise(conf.getParameter<bool>("Noise")),
      RealPedestals(conf.getParameter<bool>("RealPedestals")),
      SingleStripNoise(conf.getParameter<bool>("SingleStripNoise")),
      CommonModeNoise(conf.getParameter<bool>("CommonModeNoise")),
      BaselineShift(conf.getParameter<bool>("BaselineShift")),
      APVSaturationFromHIP(conf.getParameter<bool>("APVSaturationFromHIP")),
      theFedAlgo(conf.getParameter<int>("FedAlgorithm")),
      zeroSuppression(conf.getParameter<bool>("ZeroSuppression")),
      theElectronPerADC(conf.getParameter<double>(peakMode ? "electronPerAdcPeak" : "electronPerAdcDec")),
      theTOFCutForPeak(conf.getParameter<double>("TOFCutForPeak")),
      theTOFCutForDeconvolution(conf.getParameter<double>("TOFCutForDeconvolution")),
      tofCut(peakMode ? theTOFCutForPeak : theTOFCutForDeconvolution),
      cosmicShift(conf.getUntrackedParameter<double>("CosmicDelayShift")),
      inefficiency(conf.getParameter<double>("Inefficiency")),
      pedOffset((unsigned int)conf.getParameter<double>("PedestalsOffset")),
      PreMixing_(conf.getParameter<bool>("PreMixingMode")),
      pdtToken_(iC.esConsumes()),
      lorentzAngleToken_(iC.esConsumes(edm::ESInputTag("", conf.getParameter<std::string>("LorentzAngle")))),
      theSiHitDigitizer(new SiHitDigitizer(conf)),
      theSiPileUpSignals(new SiPileUpSignals()),
      theSiNoiseAdder(new SiGaussianTailNoiseAdder(theThreshold)),
      theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC, PreMixing_)),
      theSiZeroSuppress(new SiStripFedZeroSuppression(theFedAlgo, &iC)),
      APVProbabilityFile(conf.getParameter<edm::FileInPath>("APVProbabilityFile")),
      includeAPVSimulation_(conf.getParameter<bool>("includeAPVSimulation")),
      apv_maxResponse_(conf.getParameter<double>("apv_maxResponse")),
      apv_rate_(conf.getParameter<double>("apv_rate")),
      apv_mVPerQ_(conf.getParameter<double>("apv_mVPerQ")),
      apv_fCPerElectron_(conf.getParameter<double>("apvfCPerElectron")) {
  if (peakMode) {
    LogDebug("StripDigiInfo") << "APVs running in peak mode (poor time resolution)";
  } else {
    LogDebug("StripDigiInfo") << "APVs running in deconvolution mode (good time resolution)";
  };
  if (SingleStripNoise)
    LogDebug("SiStripDigitizerAlgorithm") << " SingleStripNoise: ON";
  else
    LogDebug("SiStripDigitizerAlgorithm") << " SingleStripNoise: OFF";
  if (CommonModeNoise)
    LogDebug("SiStripDigitizerAlgorithm") << " CommonModeNoise: ON";
  else
    LogDebug("SiStripDigitizerAlgorithm") << " CommonModeNoise: OFF";
  if (PreMixing_ && APVSaturationFromHIP)
    throw cms::Exception("PreMixing does not work with HIP loss simulation yet");
  if (APVSaturationFromHIP) {
    std::string line;
    APVProbaFile.open((APVProbabilityFile.fullPath()).c_str());
    if (APVProbaFile.is_open()) {
      while (getline(APVProbaFile, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        if (strs.size() == 2) {
          mapOfAPVprobabilities[std::stoi(strs.at(0))] = std::stof(strs.at(1));
        }
      }
      APVProbaFile.close();
    } else
      throw cms::Exception("MissingInput") << "It seems that the APV probability list is missing\n";
  }
}

SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm() {}

void SiStripDigitizerAlgorithm::initializeDetUnit(StripGeomDetUnit const* det, const SiStripBadStrip& deadChannel) {
  unsigned int detId = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();

  SiStripBadStrip::Range detBadStripRange = deadChannel.getRange(detId);
  //storing the bad strip of the the module. the module is not removed but just signal put to 0
  std::vector<bool>& badChannels = allBadChannels[detId];
  badChannels.clear();
  badChannels.insert(badChannels.begin(), numStrips, false);
  for (SiStripBadStrip::ContainerIterator it = detBadStripRange.first; it != detBadStripRange.second; ++it) {
    SiStripBadStrip::data fs = deadChannel.decode(*it);
    for (int strip = fs.firstStrip; strip < fs.firstStrip + fs.range; ++strip) {
      badChannels[strip] = true;
    }
  }
  firstChannelsWithSignal[detId] = numStrips;
  lastChannelsWithSignal[detId] = 0;

  //  if(APVSaturationFromHIP){
  //  std::bitset<6> &bs=SiStripTrackerAffectedAPVMap[detId];
  //  if(bs.any())theAffectedAPVvector.push_back(std::make_pair(detId,bs));
  //}
}

void SiStripDigitizerAlgorithm::initializeEvent(const edm::EventSetup& iSetup) {
  theSiPileUpSignals->reset();
  // This should be clear by after all calls to digitize(), but I might as well make sure
  associationInfoForDetId_.clear();

  APVSaturationProb_ = APVSaturationProbScaling_;  // reset probability
  SiStripTrackerAffectedAPVMap.clear();
  FirstLumiCalc_ = true;
  FirstDigitize_ = true;

  nTruePU_ = 0;

  //get gain noise pedestal lorentzAngle from ES handle
  setParticleDataTable(&iSetup.getData(pdtToken_));
  lorentzAngleHandle = iSetup.getHandle(lorentzAngleToken_);
}

//  Run the algorithm for a given module
//  ------------------------------------

void SiStripDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                                  std::vector<PSimHit>::const_iterator inputEnd,
                                                  size_t inputBeginGlobalIndex,
                                                  unsigned int tofBin,
                                                  const StripGeomDetUnit* det,
                                                  const GlobalVector& bfield,
                                                  const TrackerTopology* tTopo,
                                                  CLHEP::HepRandomEngine* engine) {
  // produce SignalPoints for all SimHits in detector
  unsigned int detID = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();

  size_t thisFirstChannelWithSignal = numStrips;
  size_t thisLastChannelWithSignal = 0;

  float langle = (lorentzAngleHandle.isValid()) ? lorentzAngleHandle->getLorentzAngle(detID) : 0.;

  std::vector<float> locAmpl(numStrips, 0.);

  // Loop over hits

  uint32_t detId = det->geographicalId().rawId();
  // First: loop on the SimHits
  if (CLHEP::RandFlat::shoot(engine) > inefficiency) {
    AssociationInfoForChannel* pDetIDAssociationInfo;  // I only need this if makeDigiSimLinks_ is true...
    if (makeDigiSimLinks_)
      pDetIDAssociationInfo = &(associationInfoForDetId_[detId]);  // ...so only search the map if that is the case
    std::vector<float>
        previousLocalAmplitude;  // Only used if makeDigiSimLinks_ is true. Needed to work out the change in amplitude.

    size_t simHitGlobalIndex = inputBeginGlobalIndex;  // This needs to stored to create the digi-sim link later
    for (std::vector<PSimHit>::const_iterator simHitIter = inputBegin; simHitIter != inputEnd;
         ++simHitIter, ++simHitGlobalIndex) {
      // skip hits not in this detector.
      if ((*simHitIter).detUnitId() != detId) {
        continue;
      }
      // check TOF
      if (std::fabs(simHitIter->tof() - cosmicShift -
                    det->surface().toGlobal(simHitIter->localPosition()).mag() / 30.) < tofCut &&
          simHitIter->energyLoss() > 0) {
        if (makeDigiSimLinks_)
          previousLocalAmplitude = locAmpl;  // Not needed except to make the sim link association.
        size_t localFirstChannel = numStrips;
        size_t localLastChannel = 0;
        // process the hit
        theSiHitDigitizer->processHit(
            &*simHitIter, *det, bfield, langle, locAmpl, localFirstChannel, localLastChannel, tTopo, engine);

        if (thisFirstChannelWithSignal > localFirstChannel)
          thisFirstChannelWithSignal = localFirstChannel;
        if (thisLastChannelWithSignal < localLastChannel)
          thisLastChannelWithSignal = localLastChannel;

        if (makeDigiSimLinks_) {  // No need to do any of this if truth association was turned off in the configuration
          for (size_t stripIndex = 0; stripIndex < locAmpl.size(); ++stripIndex) {
            // Work out the amplitude from this SimHit from the difference of what it was before and what it is now
            float signalFromThisSimHit = locAmpl[stripIndex] - previousLocalAmplitude[stripIndex];
            if (signalFromThisSimHit != 0) {  // If this SimHit had any contribution I need to record it.
              auto& associationVector = (*pDetIDAssociationInfo)[stripIndex];
              bool addNewEntry = true;
              // Make sure the hit isn't in already. I've seen this a few times, it always seems to happen in pairs so I think
              // it's something to do with the stereo strips.
              for (auto& associationInfo : associationVector) {
                if (associationInfo.trackID == simHitIter->trackId() &&
                    associationInfo.eventID == simHitIter->eventId()) {
                  // The hit is already in, so add this second contribution and move on
                  associationInfo.contributionToADC += signalFromThisSimHit;
                  addNewEntry = false;
                  break;
                }
              }  // end of loop over associationVector
              // If the hit wasn't already in create a new association info structure.
              if (addNewEntry)
                associationVector.push_back(AssociationInfo{
                    simHitIter->trackId(), simHitIter->eventId(), signalFromThisSimHit, simHitGlobalIndex, tofBin});
            }  // end of "if( signalFromThisSimHit!=0 )"
          }    // end of loop over locAmpl strips
        }      // end of "if( makeDigiSimLinks_ )"
      }        // end of TOF check
    }          // end for
  }
  theSiPileUpSignals->add(detID, locAmpl, thisFirstChannelWithSignal, thisLastChannelWithSignal);

  if (firstChannelsWithSignal[detID] > thisFirstChannelWithSignal)
    firstChannelsWithSignal[detID] = thisFirstChannelWithSignal;
  if (lastChannelsWithSignal[detID] < thisLastChannelWithSignal)
    lastChannelsWithSignal[detID] = thisLastChannelWithSignal;
}

//============================================================================
void SiStripDigitizerAlgorithm::calculateInstlumiScale(PileupMixingContent* puInfo) {
  //Instlumi scalefactor calculating for dynamic inefficiency

  if (puInfo && FirstLumiCalc_) {
    const std::vector<int>& bunchCrossing = puInfo->getMix_bunchCrossing();
    const std::vector<float>& TrueInteractionList = puInfo->getMix_TrueInteractions();
    const int bunchSpacing = puInfo->getMix_bunchSpacing();

    double RevFreq = 11245.;
    double minBXsec = 70.0E-27;  // use 70mb as an approximation
    double Bunch = 2100.;        // 2016 value
    if (bunchSpacing == 50)
      Bunch = Bunch / 2.;

    int pui = 0, p = 0;
    std::vector<int>::const_iterator pu;
    std::vector<int>::const_iterator pu0 = bunchCrossing.end();

    for (pu = bunchCrossing.begin(); pu != bunchCrossing.end(); ++pu) {
      if (*pu == 0) {
        pu0 = pu;
        p = pui;
      }
      pui++;
    }
    if (pu0 != bunchCrossing.end()) {  // found the in-time interaction
      nTruePU_ = TrueInteractionList.at(p);
      double instLumi = Bunch * nTruePU_ * RevFreq / minBXsec;
      APVSaturationProb_ = instLumi / 6.0E33;
    }
    FirstLumiCalc_ = false;
  }
}

//============================================================================

void SiStripDigitizerAlgorithm::digitize(edm::DetSet<SiStripDigi>& outdigi,
                                         edm::DetSet<SiStripRawDigi>& outrawdigi,
                                         edm::DetSet<SiStripRawDigi>& outStripAmplitudes,
                                         edm::DetSet<SiStripRawDigi>& outStripAmplitudesPostAPV,
                                         edm::DetSet<SiStripRawDigi>& outStripAPVBaselines,
                                         edm::DetSet<StripDigiSimLink>& outLink,
                                         const StripGeomDetUnit* det,
                                         const SiStripGain& gain,
                                         const SiStripThreshold& threshold,
                                         const SiStripNoises& noiseObj,
                                         const SiStripPedestals& pedestal,
                                         bool simulateAPVInThisEvent,
                                         const SiStripApvSimulationParameters* apvSimulationParameters,
                                         std::vector<std::pair<int, std::bitset<6>>>& theAffectedAPVvector,
                                         CLHEP::HepRandomEngine* engine,
                                         const TrackerTopology* tTopo) {
  unsigned int detID = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();

  DetId detId(detID);
  uint32_t SubDet = detId.subdetId();

  const SiPileUpSignals::SignalMapType* theSignal(theSiPileUpSignals->getSignal(detID));

  std::vector<float> detAmpl(numStrips, 0.);
  if (theSignal) {
    for (const auto& amp : *theSignal) {
      detAmpl[amp.first] = amp.second;
    }
  }

  //removing signal from the dead (and HIP effected) strips
  std::vector<bool>& badChannels = allBadChannels[detID];
  for (int strip = 0; strip < numStrips; ++strip) {
    if (badChannels[strip]) {
      detAmpl[strip] = 0.;
    }
  }

  if (includeAPVSimulation_ && simulateAPVInThisEvent) {
    // Get index in apv baseline distributions corresponding to z of detSet and PU
    const StripTopology* topol = dynamic_cast<const StripTopology*>(&(det->specificTopology()));
    LocalPoint localPos = topol->localPosition(0);
    GlobalPoint globalPos = det->surface().toGlobal(Local3DPoint(localPos.x(), localPos.y(), localPos.z()));
    float detSet_z = fabs(globalPos.z());
    float detSet_r = globalPos.perp();

    // Store SCD, before APV sim
    outStripAmplitudes.reserve(numStrips);
    for (int strip = 0; strip < numStrips; ++strip) {
      outStripAmplitudes.emplace_back(SiStripRawDigi(detAmpl[strip] / theElectronPerADC));
    }

    // Simulate APV response for each strip
    for (int strip = 0; strip < numStrips; ++strip) {
      if (detAmpl[strip] > 0) {
        // Convert charge from electrons to fC
        double stripCharge = detAmpl[strip] * apv_fCPerElectron_;

        // Get APV baseline
        double baselineV = 0;
        if (SubDet == SiStripSubdetector::TIB) {
          baselineV = apvSimulationParameters->sampleTIB(tTopo->tibLayer(detId), detSet_z, nTruePU_, engine);
        } else if (SubDet == SiStripSubdetector::TOB) {
          baselineV = apvSimulationParameters->sampleTOB(tTopo->tobLayer(detId), detSet_z, nTruePU_, engine);
        } else if (SubDet == SiStripSubdetector::TID) {
          baselineV = apvSimulationParameters->sampleTID(tTopo->tidWheel(detId), detSet_r, nTruePU_, engine);
        } else if (SubDet == SiStripSubdetector::TEC) {
          baselineV = apvSimulationParameters->sampleTEC(tTopo->tecWheel(detId), detSet_r, nTruePU_, engine);
        }
        // Store APV baseline for this strip
        outStripAPVBaselines.emplace_back(SiStripRawDigi(baselineV));

        // Fitted parameters from G Hall/M Raymond
        double maxResponse = apv_maxResponse_;
        double rate = apv_rate_;

        double outputChargeInADC = 0;
        if (baselineV < apv_maxResponse_) {
          // Convert V0 into baseline charge
          double baselineQ = -1.0 * rate * log(2 * maxResponse / (baselineV + maxResponse) - 1);

          // Add charge deposited in this BX
          double newStripCharge = baselineQ + stripCharge;

          // Apply APV response
          double signalV = 2 * maxResponse / (1 + exp(-1.0 * newStripCharge / rate)) - maxResponse;
          double gain = signalV - baselineV;

          // Convert gain (mV) to charge (assuming linear region of APV) and then to electrons
          double outputCharge = gain / apv_mVPerQ_;
          outputChargeInADC = outputCharge / apv_fCPerElectron_;
        }

        // Output charge back to original container
        detAmpl[strip] = outputChargeInADC;
      }
    }

    // Store SCD, after APV sim
    outStripAmplitudesPostAPV.reserve(numStrips);
    for (int strip = 0; strip < numStrips; ++strip)
      outStripAmplitudesPostAPV.emplace_back(SiStripRawDigi(detAmpl[strip] / theElectronPerADC));
  }

  if (APVSaturationFromHIP) {
    //Implementation of the proper charge scaling function. Need consider resaturation effect:
    //The probability map gives  the probability that at least one HIP happened during the last N bunch crossings (cfr APV recovery time).
    //The impact on the charge depends on the clostest HIP occurance (in terms of bunch crossing).
    //The function discribing the APV recovery is therefore the weighted average function which takes into account all possibilities of HIP occurances across the last bx's.

    // do this step here because we now have access to luminosity information
    if (FirstDigitize_) {
      for (std::map<int, float>::iterator iter = mapOfAPVprobabilities.begin(); iter != mapOfAPVprobabilities.end();
           ++iter) {
        std::bitset<6> bs;
        for (int Napv = 0; Napv < 6; Napv++) {
          float cursor = CLHEP::RandFlat::shoot(engine);
          bs[Napv] = cursor < iter->second * APVSaturationProb_
                         ? true
                         : false;  //APVSaturationProb has been scaled by PU luminosity
        }
        SiStripTrackerAffectedAPVMap[iter->first] = bs;
      }

      NumberOfBxBetweenHIPandEvent = 1e3;
      bool HasAtleastOneAffectedAPV = false;
      while (!HasAtleastOneAffectedAPV) {
        for (int bx = floor(300.0 / 25.0); bx > 0; bx--) {  //Reminder: make these numbers not hard coded!!
          float temp = CLHEP::RandFlat::shoot(engine) < 0.5 ? 1 : 0;
          if (temp == 1 && bx < NumberOfBxBetweenHIPandEvent) {
            NumberOfBxBetweenHIPandEvent = bx;
            HasAtleastOneAffectedAPV = true;
          }
        }
      }

      FirstDigitize_ = false;
    }

    std::bitset<6>& bs = SiStripTrackerAffectedAPVMap[detID];

    if (bs.any()) {
      // store this information so it can be saved to the event later
      theAffectedAPVvector.push_back(std::make_pair(detID, bs));

      if (!PreMixing_) {
        // Here below is the scaling function which describes the evolution of the baseline (i.e. how the charge is suppressed).
        // This must be replaced as soon as we have a proper modeling of the baseline evolution from VR runs
        float Shift =
            1 - NumberOfBxBetweenHIPandEvent / floor(300.0 / 25.0);  //Reminder: make these numbers not hardcoded!!
        float randomX = CLHEP::RandFlat::shoot(engine);
        float scalingValue = (randomX - Shift) * 10.0 / 7.0 - 3.0 / 7.0;

        for (int strip = 0; strip < numStrips; ++strip) {
          if (!badChannels[strip] && bs[strip / 128] == 1) {
            detAmpl[strip] *= scalingValue > 0 ? scalingValue : 0.0;
          }
        }
      }
    }
  }

  SiStripNoises::Range detNoiseRange = noiseObj.getRange(detID);
  SiStripApvGain::Range detGainRange = gain.getRange(detID);
  SiStripPedestals::Range detPedestalRange = pedestal.getRange(detID);

  // -----------------------------------------------------------

  auto& firstChannelWithSignal = firstChannelsWithSignal[detID];
  auto& lastChannelWithSignal = lastChannelsWithSignal[detID];
  auto iAssociationInfoByChannel =
      associationInfoForDetId_.find(detID);  // Use an iterator so that I can easily remove it once finished

  if (zeroSuppression) {
    //Adding the strip noise
    //------------------------------------------------------
    if (noise) {
      if (SingleStripNoise) {
        //      std::cout<<"In SSN, detId="<<detID<<std::endl;
        std::vector<float> noiseRMSv;
        noiseRMSv.clear();
        noiseRMSv.insert(noiseRMSv.begin(), numStrips, 0.);
        for (int strip = 0; strip < numStrips; ++strip) {
          if (!badChannels[strip]) {
            float gainValue = gain.getStripGain(strip, detGainRange);
            noiseRMSv[strip] = (noiseObj.getNoise(strip, detNoiseRange)) * theElectronPerADC / gainValue;
            //std::cout<<"<SiStripDigitizerAlgorithm::digitize>: gainValue: "<<gainValue<<"\tnoiseRMSv["<<strip<<"]: "<<noiseRMSv[strip]<<std::endl;
          }
        }
        theSiNoiseAdder->addNoiseVR(detAmpl, noiseRMSv, engine);
      } else {
        int RefStrip = int(numStrips / 2.);
        while (RefStrip < numStrips &&
               badChannels[RefStrip]) {  //if the refstrip is bad, I move up to when I don't find it
          RefStrip++;
        }
        if (RefStrip < numStrips) {
          float RefgainValue = gain.getStripGain(RefStrip, detGainRange);
          float RefnoiseRMS = noiseObj.getNoise(RefStrip, detNoiseRange) * theElectronPerADC / RefgainValue;

          theSiNoiseAdder->addNoise(
              detAmpl, firstChannelWithSignal, lastChannelWithSignal, numStrips, RefnoiseRMS, engine);
          //std::cout<<"<SiStripDigitizerAlgorithm::digitize>: RefgainValue: "<<RefgainValue<<"\tRefnoiseRMS: "<<RefnoiseRMS<<std::endl;
        }
      }
    }  //if noise

    DigitalVecType digis;
    theSiZeroSuppress->suppress(
        theSiDigitalConverter->convert(detAmpl, &gain, detID), digis, detID, noiseObj, threshold);
    // Now do the association to truth. Note that if truth association was turned off in the configuration this map
    // will be empty and the iterator will always equal associationInfoForDetId_.end().
    if (iAssociationInfoByChannel !=
        associationInfoForDetId_.end()) {  // make sure the readings for this DetID aren't completely from noise
      for (const auto& iDigi : digis) {
        auto& associationInfoByChannel = iAssociationInfoByChannel->second;
        const std::vector<AssociationInfo>& associationInfo = associationInfoByChannel[iDigi.channel()];

        // Need to find the total from all sim hits, because this might not be the same as the total
        // digitised due to noise or whatever.
        float totalSimADC = 0;
        for (const auto& iAssociationInfo : associationInfo)
          totalSimADC += iAssociationInfo.contributionToADC;
        // Now I know that I can loop again and create the links
        for (const auto& iAssociationInfo : associationInfo) {
          // Note simHitGlobalIndex used to have +1 because TrackerHitAssociator (the only place I can find this value being used)
          // expected counting to start at 1, not 0.  Now changed.
          outLink.push_back(StripDigiSimLink(iDigi.channel(),
                                             iAssociationInfo.trackID,
                                             iAssociationInfo.simHitGlobalIndex,
                                             iAssociationInfo.tofBin,
                                             iAssociationInfo.eventID,
                                             iAssociationInfo.contributionToADC / totalSimADC));
        }  // end of loop over associationInfo
      }    // end of loop over the digis
    }      // end of check that iAssociationInfoByChannel is a valid iterator
    outdigi.data = digis;
  }  //if zeroSuppression

  if (!zeroSuppression) {
    //if(noise){
    // the constant pedestal offset is needed because
    //   negative adc counts are not allowed in case
    //   Pedestal and CMN subtraction is performed.
    //   The pedestal value read from the conditions
    //   is pedValue and after the pedestal subtraction
    //   the baseline is zero. The Common Mode Noise
    //   is not subtracted from the negative adc counts
    //   channels. Adding pedOffset the baseline is set
    //   to pedOffset after pedestal subtraction and CMN
    //   is subtracted to all the channels since none of
    //   them has negative adc value. The pedOffset is
    //   treated as a constant component in the CMN
    //   estimation and subtracted as CMN.

    //calculating the charge deposited on each APV and subtracting the shift
    //------------------------------------------------------
    if (BaselineShift) {
      theSiNoiseAdder->addBaselineShift(detAmpl, badChannels);
    }

    //Adding the strip noise
    //------------------------------------------------------
    if (noise) {
      std::vector<float> noiseRMSv;
      noiseRMSv.clear();
      noiseRMSv.insert(noiseRMSv.begin(), numStrips, 0.);

      if (SingleStripNoise) {
        for (int strip = 0; strip < numStrips; ++strip) {
          if (!badChannels[strip])
            noiseRMSv[strip] = (noiseObj.getNoise(strip, detNoiseRange)) * theElectronPerADC;
        }

      } else {
        int RefStrip = 0;  //int(numStrips/2.);
        while (RefStrip < numStrips &&
               badChannels[RefStrip]) {  //if the refstrip is bad, I move up to when I don't find it
          RefStrip++;
        }
        if (RefStrip < numStrips) {
          float noiseRMS = noiseObj.getNoise(RefStrip, detNoiseRange) * theElectronPerADC;
          for (int strip = 0; strip < numStrips; ++strip) {
            if (!badChannels[strip])
              noiseRMSv[strip] = noiseRMS;
          }
        }
      }

      theSiNoiseAdder->addNoiseVR(detAmpl, noiseRMSv, engine);
    }

    //adding the CMN
    //------------------------------------------------------
    if (CommonModeNoise) {
      float cmnRMS = 0.;
      DetId detId(detID);
      switch (detId.subdetId()) {
        case SiStripSubdetector::TIB:
          cmnRMS = cmnRMStib;
          break;
        case SiStripSubdetector::TID:
          cmnRMS = cmnRMStid;
          break;
        case SiStripSubdetector::TOB:
          cmnRMS = cmnRMStob;
          break;
        case SiStripSubdetector::TEC:
          cmnRMS = cmnRMStec;
          break;
      }
      cmnRMS *= theElectronPerADC;
      theSiNoiseAdder->addCMNoise(detAmpl, cmnRMS, badChannels, engine);
    }

    //Adding the pedestals
    //------------------------------------------------------

    std::vector<float> vPeds;
    vPeds.clear();
    vPeds.insert(vPeds.begin(), numStrips, 0.);

    if (RealPedestals) {
      for (int strip = 0; strip < numStrips; ++strip) {
        if (!badChannels[strip])
          vPeds[strip] = (pedestal.getPed(strip, detPedestalRange) + pedOffset) * theElectronPerADC;
      }
    } else {
      for (int strip = 0; strip < numStrips; ++strip) {
        if (!badChannels[strip])
          vPeds[strip] = pedOffset * theElectronPerADC;
      }
    }

    theSiNoiseAdder->addPedestals(detAmpl, vPeds);

    //if(!RealPedestals&&!CommonModeNoise&&!noise&&!BaselineShift&&!APVSaturationFromHIP){

    //  edm::LogWarning("SiStripDigitizer")<<"You are running the digitizer without Noise generation and without applying Zero Suppression. ARE YOU SURE???";
    //}else{

    DigitalRawVecType rawdigis = theSiDigitalConverter->convertRaw(detAmpl, &gain, detID);

    // Now do the association to truth. Note that if truth association was turned off in the configuration this map
    // will be empty and the iterator will always equal associationInfoForDetId_.end().
    if (iAssociationInfoByChannel !=
        associationInfoForDetId_.end()) {  // make sure the readings for this DetID aren't completely from noise
      // N.B. For the raw digis the channel is inferred from the position in the vector.
      // I'VE NOT TESTED THIS YET!!!!!
      // ToDo Test this properly.
      for (size_t channel = 0; channel < rawdigis.size(); ++channel) {
        auto& associationInfoByChannel = iAssociationInfoByChannel->second;
        const auto iAssociationInfo = associationInfoByChannel.find(channel);
        if (iAssociationInfo == associationInfoByChannel.end())
          continue;  // Skip if there is no sim information for this channel (i.e. it's all noise)
        const std::vector<AssociationInfo>& associationInfo = iAssociationInfo->second;

        // Need to find the total from all sim hits, because this might not be the same as the total
        // digitised due to noise or whatever.
        float totalSimADC = 0;
        for (const auto& iAssociationInfo : associationInfo)
          totalSimADC += iAssociationInfo.contributionToADC;
        // Now I know that I can loop again and create the links
        for (const auto& iAssociationInfo : associationInfo) {
          // Note simHitGlobalIndex used to have +1 because TrackerHitAssociator (the only place I can find this value being used)
          // expected counting to start at 1, not 0.  Now changed.
          outLink.push_back(StripDigiSimLink(channel,
                                             iAssociationInfo.trackID,
                                             iAssociationInfo.simHitGlobalIndex,
                                             iAssociationInfo.tofBin,
                                             iAssociationInfo.eventID,
                                             iAssociationInfo.contributionToADC / totalSimADC));
        }  // end of loop over associationInfo
      }    // end of loop over the digis
    }      // end of check that iAssociationInfoByChannel is a valid iterator

    outrawdigi.data = rawdigis;

    //}
  }

  // Now that I've finished with this entry in the map of associations, I can remove it.
  // Note that there might not be an association if the ADC reading is from noise in which
  // case associationIsValid will be false.
  if (iAssociationInfoByChannel != associationInfoForDetId_.end())
    associationInfoForDetId_.erase(iAssociationInfoByChannel);
}
