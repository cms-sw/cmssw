// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.

#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "CLHEP/Random/RandFlat.h"

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng):
  conf_(conf),rndEngine(eng){
  theThreshold              = conf_.getParameter<double>("NoiseSigmaThreshold");
  theElectronPerADC         = conf_.getParameter<double>("electronPerAdc");
  theFedAlgo                = conf_.getParameter<int>("FedAlgorithm");
  peakMode                  = conf_.getParameter<bool>("APVpeakmode");
  noise                     = conf_.getParameter<bool>("Noise");
  zeroSuppression           = conf_.getParameter<bool>("ZeroSuppression");
  theTOFCutForPeak          = conf_.getParameter<double>("TOFCutForPeak");
  theTOFCutForDeconvolution = conf_.getParameter<double>("TOFCutForDeconvolution");
  cosmicShift               = conf_.getUntrackedParameter<double>("CosmicDelayShift");
  inefficiency              = conf_.getParameter<double>("Inefficiency");
  
  if (peakMode) {
    tofCut=theTOFCutForPeak;
    LogDebug("StripDigiInfo")<<"APVs running in peak mode (poor time resolution)";
  } else {
    tofCut=theTOFCutForDeconvolution;
    LogDebug("StripDigiInfo")<<"APVs running in deconvolution mode (good time resolution)";
  };
  
  theSiHitDigitizer = new SiHitDigitizer(conf_,rndEngine);
  theSiPileUpSignals = new SiPileUpSignals();
  theSiNoiseAdder = new SiGaussianTailNoiseAdder(theThreshold,rndEngine);
  theSiDigitalConverter = new SiTrivialDigitalConverter(theElectronPerADC);
  theSiZeroSuppress = new SiStripFedZeroSuppression(theFedAlgo);
}

SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){
  delete theSiHitDigitizer;
  delete theSiPileUpSignals;
  delete theSiNoiseAdder;
  delete theSiDigitalConverter;
  delete theSiZeroSuppress;
}

//  Run the algorithm for a given module
//  ------------------------------------

void SiStripDigitizerAlgorithm::run(edm::DetSet<SiStripDigi>& outdigi,
				    edm::DetSet<SiStripRawDigi>& outrawdigi,
				    const std::vector<std::pair<PSimHit, int > > &input,
				    StripGeomDetUnit *det,
				    GlobalVector bfield,float langle, 
				    edm::ESHandle<SiStripGain> & gainHandle,
				    edm::ESHandle<SiStripThreshold> & thresholdHandle,
				    edm::ESHandle<SiStripNoises> & noiseHandle,
				    edm::ESHandle<SiStripPedestals> & pedestalHandle
				   ) {  
  theSiPileUpSignals->reset();
  unsigned int detID = det->geographicalId().rawId();
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(detID);
  numStrips = (det->specificTopology()).nstrips();
  strip = int(numStrips/2.);
  // WARNING!!!
  // here only the value for noise and pedestal from the central strip are taken
  //    in case the noise RMS / pedestal values are not flat
  //    the value strip-by-strip must be taken in SiGaussianTailNoiseAdded
  noiseRMS = noiseHandle->getNoise(strip,detNoiseRange);
  pedValue = pedestalHandle->getPed(strip,detPedestalRange);

  // local amplitude of detector channels (from processed PSimHit)
  std::vector<double> locAmpl(numStrips,0.);
  // total amplitude of detector channels
  std::vector<double> detAmpl(locAmpl);

  // to speed-up vector filling (only for channels with signal)
  //  the unsigned int corresponds to the channel number
  //  and NOT to the vector position (which is channel-1)
  unsigned int firstChannelWithSignal = numStrips+1;
  unsigned int lastChannelWithSignal  = 0;

  // First: loop on the SimHits
  std::vector<std::pair<PSimHit, int > >::const_iterator simHitIter = input.begin();
  std::vector<std::pair<PSimHit, int > >::const_iterator simHitIterEnd = input.end();
  if(RandFlat::shoot()>inefficiency) {
    for (;simHitIter != simHitIterEnd; ++simHitIter) {
      // check TOF
      const PSimHit & ihit = (*simHitIter).first;
      float t0 = det->surface().toGlobal(ihit.localPosition()).mag()/30.;
      if ( std::fabs( ihit.tof() - cosmicShift - t0) < tofCut && ihit.energyLoss()>0) {
        unsigned int localFirstChannel = numStrips+1;
        unsigned int localLastChannel  = 0;
        // process the hit
        theSiHitDigitizer->processHit(ihit,*det,bfield,langle, locAmpl, localFirstChannel, localLastChannel);
        theSiPileUpSignals->add(locAmpl, localFirstChannel, localLastChannel, ihit, (*simHitIter).second);
        // sum signal on strips
        for (unsigned int iChannel=localFirstChannel; iChannel<=localLastChannel; iChannel++) {
          if(locAmpl[iChannel]>0.) {             
            detAmpl[iChannel]+=locAmpl[iChannel];
            locAmpl[iChannel]=0;
           }
        }
        if(firstChannelWithSignal>localFirstChannel) firstChannelWithSignal=localFirstChannel;
        if(lastChannelWithSignal<localLastChannel) lastChannelWithSignal=localLastChannel;
      }
    }
  }
  
  const SiPileUpSignals::HitToDigisMapType& theLink = theSiPileUpSignals->dumpLink();  
  const SiPileUpSignals::HitCounterToDigisMapType& theCounterLink = theSiPileUpSignals->dumpCounterLink();  
  
  if(zeroSuppression){
    if(noise) 
      theSiNoiseAdder->addNoise(detAmpl,firstChannelWithSignal,lastChannelWithSignal,numStrips,noiseRMS*theElectronPerADC);
    digis.clear();
    theSiZeroSuppress->suppress(theSiDigitalConverter->convert(detAmpl, gainHandle, detID), digis, detID,noiseHandle,thresholdHandle);
    push_link(digis, theLink, theCounterLink, detAmpl,detID);
    outdigi.data = digis;
  }
  
  if(!zeroSuppression){
    if(noise){
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
      float pedOffset = 128.; // ADC counts
      pedValue += pedOffset;
      theSiNoiseAdder->createRaw(detAmpl,firstChannelWithSignal,lastChannelWithSignal,
                                 numStrips,noiseRMS*theElectronPerADC,pedValue*theElectronPerADC);
    } else {
      edm::LogWarning("SiStripDigitizer")<<"You are running the digitizer without Noise generation and without applying Zero Suppression. ARE YOU SURE???";
    }
    rawdigis.clear();
    rawdigis = theSiDigitalConverter->convertRaw(detAmpl, gainHandle, detID);
    push_link_raw(rawdigis, theLink, theCounterLink, detAmpl,detID);
    outrawdigi.data = rawdigis;
  }
}

void SiStripDigitizerAlgorithm::push_link(const DigitalVecType &digis,
					  const HitToDigisMapType& htd,
					  const HitCounterToDigisMapType& hctd,
					  const std::vector<double>& afterNoise,
					  unsigned int detID) {
  link_coll.clear();  
  for ( DigitalVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(i->strip()));  
    if (mi == htd.end()) continue;
    HitCounterToDigisMapType::const_iterator cmi(hctd.find(i->strip()));  
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    double totalAmplitude1 = afterNoise[(*mi).first];
    
    //--- digisimlink
    int sim_counter=0; 
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      float threshold = 0.;
      float fraction = (*iter).second/totalAmplitude1;
      if ( fraction >= threshold) {
	// Noise fluctuation could make fraction>1. Unphysical, set it by hand = 1.
	if(fraction > 1.) fraction = 1.;
	for (std::vector < std::pair < const PSimHit*, int > >::const_iterator 
	       simcount = (*cmi).second.begin() ; simcount != (*cmi).second.end(); simcount ++){
	  if((*iter).first == (*simcount).first) sim_counter = (*simcount).second;
	}
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      sim_counter, //simhit counter
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}

void SiStripDigitizerAlgorithm::push_link_raw(const DigitalRawVecType &digis,
					      const HitToDigisMapType& htd,
					      const HitCounterToDigisMapType& hctd,
					      const std::vector<double>& afterNoise,
					      unsigned int detID) {
  link_coll.clear();  
  int nstrip = -1;
  for ( DigitalRawVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    nstrip++;
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(nstrip));  
    HitCounterToDigisMapType::const_iterator cmi(hctd.find(nstrip));  
    if (mi == htd.end()) continue;
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    double totalAmplitude1 = afterNoise[(*mi).first];
    
    //--- digisimlink
    int sim_counter_raw=0;
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      float threshold = 0.;
      float fraction = (*iter).second/totalAmplitude1;
      if (fraction >= threshold) {
	//Noise fluctuation could make fraction>1. Unphysical, set it by hand.
	if(fraction >1.) fraction = 1.;
	//add counter information
	for (std::vector < std::pair < const PSimHit*, int > >::const_iterator 
	       simcount = (*cmi).second.begin() ; simcount != (*cmi).second.end(); simcount ++){
	  if((*iter).first == (*simcount).first) sim_counter_raw = (*simcount).second;
	}
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      sim_counter_raw, //simhit counter
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}
