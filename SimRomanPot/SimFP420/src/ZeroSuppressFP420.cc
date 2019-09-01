///////////////////////////////////////////////////////////////////////////////
// File: ZeroSuppressFP420.cc
// Date: 12.2006
// Description: ZeroSuppressFP420 for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimRomanPot/SimFP420/interface/ZeroSuppressFP420.h"

ZeroSuppressFP420::ZeroSuppressFP420(const edm::ParameterSet &conf, float noise) : conf_(conf), theNumFEDalgos(4) {
  noiseInAdc = noise;
  initParams(conf_);
  verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
  // initParams();
  if (verbosity > 0) {
    std::cout << "ZeroSuppressFP420: constructor: noiseInAdc=  " << noiseInAdc << std::endl;
  }
}

/*
 * The zero suppression algorithm, implemented in the trkFEDclusterizer method.
 * The class publically inherits from the ZSuppressFP420 class, which requires
 * the use of a method named zeroSuppress.
 */

void ZeroSuppressFP420::initParams(edm::ParameterSet const &conf_) {
  verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
  algoConf = conf_.getParameter<int>("FedFP420Algorithm");               // FedFP420Algorithm: =1  (,2,3,4)
  lowthreshConf = conf_.getParameter<double>("FedFP420LowThreshold");    // FedFP420LowThreshold  =3.
  highthreshConf = conf_.getParameter<double>("FedFP420HighThreshold");  // FedFP420HighThreshold  =4.

  /*
   * There are four possible algorithms, the default of which (4)
   * has different thresholds for isolated channels and ones in clusters.
   * It also merges clusters (single or multi channels) that are only separated
   * by one hole. This channel is selected as signal even though it is below
   * both thresholds.
   */

  theFEDalgorithm = algoConf;
  theFEDlowThresh = lowthreshConf * noiseInAdc;
  theFEDhighThresh = highthreshConf * noiseInAdc;

  if (verbosity > 0) {
    std::cout << "ZeroSuppressFP420: initParams: !!!  theFEDalgorithm=  " << theFEDalgorithm << std::endl;
    std::cout << " lowthreshConf=  " << lowthreshConf << " highthreshConf=  " << highthreshConf
              << " theFEDlowThresh=  " << theFEDlowThresh << " theFEDhighThresh=  " << theFEDhighThresh << std::endl;
  }

  // Check zero suppress algorithm
  if (theFEDalgorithm < 1 || theFEDalgorithm > theNumFEDalgos) {
    edm::LogError("FP420DigiInfo") << "ZeroSuppressFP420 FATAL ERROR: Unknown zero suppress algorithm "
                                   << theFEDalgorithm;
    exit(1);
  }

  // Check thresholds
  if (theFEDlowThresh > theFEDhighThresh) {
    edm::LogError("FP420DigiInfo") << "ZeroSuppressFP420 FATAL ERROR: Low threshold exceeds high "
                                      "threshold: "
                                   << theFEDlowThresh << " > " << theFEDhighThresh;
    exit(2);
  }
}

ZSuppressFP420::DigitalMapType ZeroSuppressFP420::zeroSuppress(const DigitalMapType &notZeroSuppressedMap, int vrb) {
  return trkFEDclusterizer(notZeroSuppressedMap, vrb);
  if (vrb > 0) {
    std::cout << "zeroSuppress: return trkFEDclusterizer(notZeroSuppressedMap)" << std::endl;
  }
}

// This performs the zero suppression
ZSuppressFP420::DigitalMapType ZeroSuppressFP420::trkFEDclusterizer(const DigitalMapType &in, int vrb) {
  const std::string s2("ZeroSuppressFP420::trkFEDclusterizer1");

  DigitalMapType selectedSignal;
  DigitalMapType::const_iterator i, iPrev, iNext, iPrev2, iNext2;

  if (vrb > 0) {
    std::cout << "Before For loop" << std::endl;
  }

  for (i = in.begin(); i != in.end(); i++) {
    // Find adc values for neighbouring strips
    int strip = i->first;
    int adc = i->second;
    iPrev = in.find(strip - 1);
    iNext = in.find(strip + 1);
    if (vrb > 0) {
      std::cout << "Inside For loop trkFEDclusterizer: strip= " << strip << " adc= " << adc << std::endl;
    }
    // Set values for channels just outside module to infinity.
    // This is to avoid losing channels at the edges,
    // which otherwise would pass cuts if strips were next to each other.
    int adcPrev = -99999;
    int adcNext = -99999;
    if (((strip) % 128) == 127)
      adcNext = 99999;
    if (((strip) % 128) == 0)
      adcPrev = 99999;
    // Otherwise if channel was found then find it's ADC count.
    if (iPrev != in.end())
      adcPrev = iPrev->second;
    if (iNext != in.end())
      adcNext = iNext->second;
    int adcMaxNeigh = std::max(adcPrev, adcNext);
    if (vrb > 0) {
      std::cout << "adcPrev= " << adcPrev << " adcNext= " << adcNext << " adcMaxNeigh= " << adcMaxNeigh << std::endl;
    }

    // Find adc values for next neighbouring channes
    iPrev2 = in.find(strip - 2);
    iNext2 = in.find(strip + 2);
    // See above
    int adcPrev2 = -99999;
    int adcNext2 = -99999;
    if (((strip) % 128) == 126)
      adcNext2 = 99999;
    if (((strip) % 128) == 1)
      adcPrev2 = 99999;
    if (iPrev2 != in.end())
      adcPrev2 = iPrev2->second;
    if (iNext2 != in.end())
      adcNext2 = iNext2->second;

    if (vrb > 0) {
      std::cout << "adcPrev2= " << adcPrev2 << " adcNext2= " << adcNext2 << std::endl;
      std::cout << "To be accepted or not?  adc= " << adc << " >= theFEDlowThresh=" << theFEDlowThresh << std::endl;
    }
    // Decide if this channel should be accepted.
    bool accept = false;
    switch (theFEDalgorithm) {
      case 1:
        accept = (adc >= theFEDlowThresh);
        break;

      case 2:
        accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh && adcMaxNeigh >= theFEDlowThresh));
        break;

      case 3:
        accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh && adcMaxNeigh >= theFEDhighThresh));
        break;

      case 4:
        accept = ((adc >= theFEDhighThresh) ||  // Test for adc>highThresh (same as
                                                // algorithm 2)
                  ((adc >= theFEDlowThresh) &&  // Test for adc>lowThresh, with neighbour
                                                // adc>lowThresh (same as algorithm 2)
                   (adcMaxNeigh >= theFEDlowThresh)) ||
                  ((adc < theFEDlowThresh) &&          // Test for adc<lowThresh
                   (((adcPrev >= theFEDhighThresh) &&  // with both neighbours>highThresh
                     (adcNext >= theFEDhighThresh)) ||
                    ((adcPrev >= theFEDhighThresh) &&  // OR with previous neighbour>highThresh and
                     (adcNext >= theFEDlowThresh) &&   // both the next neighbours>lowThresh
                     (adcNext2 >= theFEDlowThresh)) ||
                    ((adcNext >= theFEDhighThresh) &&  // OR with next neighbour>highThresh and
                     (adcPrev >= theFEDlowThresh) &&   // both the previous neighbours>lowThresh
                     (adcPrev2 >= theFEDlowThresh)) ||
                    ((adcNext >= theFEDlowThresh) &&   // OR with both next neighbours>lowThresh and
                     (adcNext2 >= theFEDlowThresh) &&  // both the previous neighbours>lowThresh
                     (adcPrev >= theFEDlowThresh) && (adcPrev2 >= theFEDlowThresh)))));
        break;
    }

    /*
     * When a channel satisfying only the lower threshold is at the edge of an
     * APV or module, the trkFEDclusterizer method assumes that every channel
     * just outside an APV or module has a hit on it. This is to avoid channel
     * inefficiencies at the edges of APVs and modules.
     */
    if (accept) {
      selectedSignal[strip] = adc;

      if (vrb > 0) {
        std::cout << "selected strips = " << strip << " adc= " << adc << std::endl;
      }
    }
  }

  if (vrb > 0) {
    std::cout << "last line of trkFEDclusterizer: return selectedSignal" << std::endl;
  }
  return selectedSignal;
}
