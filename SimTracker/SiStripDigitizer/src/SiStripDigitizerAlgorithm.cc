// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.
// Author:  A. Giammanco
// Creation Date:  Oct. 2, 2005   

#include <vector>
#include <iostream>
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  if (peakMode) {
    tofCut=theTOFCutForPeak;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      edm::LogInfo("StripDigiInfo")<<"APVs running in peak mode (poor time resolution)";
    }
  } else {
    tofCut=theTOFCutForDeconvolution;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      edm::LogInfo("StripDigiInfo")<<"APVs running in deconvolution mode (good time resolution)";
    }
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


//  Run the algorithm
//  ------------------

void SiStripDigitizerAlgorithm::run(edm::DetSet<SiStripDigi>& outdigi,
				    edm::DetSet<SiStripRawDigi>& outrawdigi,
				    const std::vector<PSimHit> &input,
				    StripGeomDetUnit *det,
				    GlobalVector bfield,float langle, 
				    edm::ESHandle<SiStripGain> & gainHandle,
				    edm::ESHandle<SiStripPedestals> & pedestalsHandle,
				    edm::ESHandle<SiStripNoises> & noiseHandle){
  
  theSiPileUpSignals->reset();
  unsigned int detID = det->geographicalId().rawId();
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  // We will work on ONE copy of the map only,
  //  and pass references where it is needed.
  signal_map_type theSignal;
  signal_map_type theSignal_forLink;

  //
  // First: loop on the SimHits
  //
  std::vector<PSimHit>::const_iterator simHitIter = input.begin();
  std::vector<PSimHit>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const PSimHit & ihit = *simHitIter;

    float dist = det->surface().toGlobal(ihit.localPosition()).mag();
    float t0 = dist/30.;  // light velocity = 30 cm/ns      

    if ( std::fabs( ihit.tof() - cosmicShift - t0) < tofCut && ihit.energyLoss()>0) {
      theSiHitDigitizer->processHit(ihit,*det,bfield,langle, theSignal,theSignal_forLink);
      theSiPileUpSignals->add(theSignal_forLink, ihit);
    }
    theSignal_forLink.clear();
  }
  
  SiPileUpSignals::HitToDigisMapType theLink = theSiPileUpSignals->dumpLink();  
  
  numStrips = (det->specificTopology()).nstrips();
  strip = int(numStrips/2.);
  noiseRMS = noiseHandle->getNoise(strip,detNoiseRange);

  if(zeroSuppression){
    if(noise) 
      theSiNoiseAdder->addNoise(theSignal,numStrips,noiseRMS*theElectronPerADC);
    digis.clear();
    theSiZeroSuppress->suppress(theSiDigitalConverter->convert(theSignal, gainHandle, detID), digis, detID,noiseHandle,pedestalsHandle);
    push_link(digis, theLink, theSignal,detID);
    outdigi.data = digis;
  }
  
  if(!zeroSuppression){
    if(noise){
      theSiNoiseAdder->createRaw(theSignal,numStrips,noiseRMS*theElectronPerADC);
    }else{
      edm::LogWarning("SiStripDigitizer")<<"You are running the digitizer without Noise generation and without applying Zero Suppression. ARE YOU SURE???";
    }
    rawdigis.clear();
    rawdigis = theSiDigitalConverter->convertRaw(theSignal, gainHandle, detID);
    push_link_raw(rawdigis, theLink, theSignal,detID);
    outrawdigi.data = rawdigis;
  }
}

void SiStripDigitizerAlgorithm::push_link(const DigitalVecType &digis,
					  const HitToDigisMapType& htd,
					  const SiPileUpSignals::signal_map_type& afterNoise,
					  unsigned int detID){
  link_coll.clear();  
  
  for ( DigitalVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(i->strip()));  
    if (mi == htd.end()) continue;
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    
    SiPileUpSignals::signal_map_type& temp1 = const_cast<SiPileUpSignals::signal_map_type&>(afterNoise);
    float totalAmplitude1 = temp1[(*mi).first];
    
    //--- digisimlink
    
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      
      float threshold = 0.;
      if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {
	float fraction = totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1;
	
	//Noise fluctuation could make fraction>1. Unphysical, set it by hand.
	if(fraction >1.) fraction = 1.;
	
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}

void SiStripDigitizerAlgorithm::push_link_raw(const DigitalRawVecType &digis,
					  const HitToDigisMapType& htd,
					  const SiPileUpSignals::signal_map_type& afterNoise,
					  unsigned int detID){
  link_coll.clear();  
  
  int nstrip = -1;
  for ( DigitalRawVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    nstrip++;
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(nstrip));  
    if (mi == htd.end()) continue;
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    
    SiPileUpSignals::signal_map_type& temp1 = const_cast<SiPileUpSignals::signal_map_type&>(afterNoise);
    float totalAmplitude1 = temp1[(*mi).first];
    
    //--- digisimlink
    
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      
      float threshold = 0.;
      if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {




	float fraction = totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1;
	
	//Noise fluctuation could make fraction>1. Unphysical, set it by hand.
	if(fraction >1.) fraction = 1.;
	
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}

void SiStripDigitizerAlgorithm::setParticleDataTable(const ParticleDataTable * pdt)
{
  theSiHitDigitizer->setParticleDataTable(pdt);
}
