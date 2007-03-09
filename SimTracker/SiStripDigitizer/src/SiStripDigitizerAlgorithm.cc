// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.
// Author:  A. Giammanco
// Creation Date:  Oct. 2, 2005   

#include <vector>
#include <iostream>
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, StripGeomDetUnit *det,
						     uint32_t& idForNoise , SiStripNoiseService* noiseService,const ParticleDataTable * pdt):conf_(conf),
																       pdt_(pdt){
  //  cout << "Creating a SiStripDigitizerAlgorithm." << endl;

  ndigis=0;
  SiStripNoiseService_=noiseService;
  NumberOfSegments = 20; // Default number of track segment divisions
  ClusterWidth = 3.; // Charge integration spread on the collection plane
  Sigma0 = 0.0007; // Charge diffusion constant 
  Dist300 = 0.0300; // normalized to 300micron Silicon
  theElectronPerADC = conf_.getParameter<double>("ElectronPerAdc");
  theThreshold      = conf_.getParameter<double>("AdcThreshold");
  theAdcFullScale   = conf_.getParameter<int>("AdcFullScale");
  noNoise           = conf_.getParameter<bool>("NoNoise");
  peakMode          = conf_.getParameter<bool>("APVpeakmode"); 
  if (peakMode) {
    tofCut=100;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      edm::LogInfo("StripDigiInfo")<<"APVs running in peak mode (poor time resolution)";
    }
  } else {
    tofCut=50;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      edm::LogInfo("StripDigiInfo")<<"APVs running in deconvolution mode (good time resolution)";
    }
  };
 
  topol = &det->specificTopology(); // cache topology
  numStrips = topol->nstrips();   // det module number of strips
  int strip = int(numStrips/2.);
  float noiseRMS = SiStripNoiseService_->getNoise(idForNoise,strip);

  theSiNoiseAdder = new SiGaussianTailNoiseAdder(numStrips,noiseRMS*theElectronPerADC,theThreshold);
  theSiZeroSuppress = new SiTrivialZeroSuppress(conf_,noiseRMS);
  theSiHitDigitizer = new SiHitDigitizer(conf_,det,pdt_);
  theSiPileUpSignals = new SiPileUpSignals();
  theSiDigitalConverter = new SiTrivialDigitalConverter(theElectronPerADC,theAdcFullScale);

  
}


SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){

  //cout << "Destroying a SiStripDigitizerAlgorithm." << endl;

  delete theSiNoiseAdder;
  delete theSiZeroSuppress;
  delete theSiHitDigitizer;
  delete theSiPileUpSignals;
  delete theSiDigitalConverter;
  
}


//  Run the algorithm
//  ------------------

edm::DetSet<SiStripDigi>::collection_type SiStripDigitizerAlgorithm::run(const std::vector<PSimHit> &input,
									 StripGeomDetUnit *det,
									 GlobalVector bfield){
  
  //  std::cout << "SiStripDigitizerAlgorithm is running!" << endl;
  
  theSiPileUpSignals->reset();
  unsigned int detID = det->geographicalId().rawId();
  
  //
  // First: loop on the SimHits
  //
  std::vector<PSimHit>::const_iterator simHitIter = input.begin();
  std::vector<PSimHit>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    
    const PSimHit& ihit = *simHitIter;
    
    if ( std::fabs(ihit.tof()) < tofCut && ihit.energyLoss()>0) {
      SiHitDigitizer::hit_map_type _temp = theSiHitDigitizer->processHit(ihit,*det,bfield);
      theSiPileUpSignals->add(_temp,ihit);
      
    }
  }
  
  SiPileUpSignals::signal_map_type theSignal = theSiPileUpSignals->dumpSignal();
  SiPileUpSignals::HitToDigisMapType theLink = theSiPileUpSignals->dumpLink();  
  SiPileUpSignals::signal_map_type afterNoise;
  if (noNoise) {
    afterNoise = theSignal;
  } else {
    afterNoise = theSiNoiseAdder->addNoise(theSignal);
  }
  
  digis.clear();
  push_digis(theSiZeroSuppress->zeroSuppress(theSiDigitalConverter->convert(afterNoise)),
	     theLink,afterNoise,detID);
  
  return digis;
}

void SiStripDigitizerAlgorithm::push_digis(const DigitalMapType& dm,
					   const HitToDigisMapType& htd,
					   const SiPileUpSignals::signal_map_type& afterNoise,
					   unsigned int detID){
  digis.reserve(50); 
  digis.clear();
  link_coll.clear();  
  
  for ( DigitalMapType::const_iterator i=dm.begin(); i!=dm.end(); i++) {
    digis.push_back( SiStripDigi( (*i).first, (*i).second));
    ndigis++; 
  }
  
  // reworked to access the fraction of amplitude per simhit
  
  for ( HitToDigisMapType::const_iterator mi=htd.begin(); mi!=htd.end(); mi++) {
    
    if ((*((const_cast<DigitalMapType * >(&dm)))).find((*mi).first) != (*((const_cast<DigitalMapType * >(&dm)))).end() ){           
      // --- For each channel, sum up the signals from a simtrack
      
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
}


