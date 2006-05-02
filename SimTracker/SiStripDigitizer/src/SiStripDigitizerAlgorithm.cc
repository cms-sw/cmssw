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
#include "Geometry/Surface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, StripGeomDetUnit *det):conf_(conf){
  //  cout << "Creating a SiStripDigitizerAlgorithm." << endl;

  ndigis=0;

  NumberOfSegments = 20;          // Default number of track segment divisions
  ClusterWidth = 3.;              // Charge integration spread on the collection plane
  Sigma0 = 0.0007;                // Charge diffusion constant 
  Dist300 = 0.0300;               // normalized to 300micron Silicon
  theElectronPerADC = conf_.getParameter<double>("ElectronPerAdc");
  theThreshold      = conf_.getParameter<double>("AdcThreshold");
  ENC               = conf_.getParameter<double>("EquivalentNoiseCharge300um");
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
  //thickness:
  const BoundSurface& p = (dynamic_cast<StripGeomDetUnit*>((det)))->surface();
  moduleThickness = p.bounds().thickness();
  float noiseRMS = ENC*moduleThickness/(0.03);

  theSiNoiseAdder = new SiGaussianTailNoiseAdder(numStrips,noiseRMS,theThreshold);
  theSiZeroSuppress = new SiTrivialZeroSuppress(conf_,noiseRMS/theElectronPerADC);
  theSiHitDigitizer = new SiHitDigitizer(conf_,det);
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
  vector<PSimHit>::const_iterator simHitIter = input.begin();
  vector<PSimHit>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    
    const PSimHit ihit = *simHitIter;
    
    if ( abs(ihit.tof()) < tofCut && ihit.energyLoss()>0) {
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
  
  digis.data.clear();
  push_digis(theSiZeroSuppress->zeroSuppress(theSiDigitalConverter->convert(afterNoise)),
	     theLink,afterNoise,detID);
  
  return digis.data;
}

void SiStripDigitizerAlgorithm::push_digis(const DigitalMapType& dm,
					   const HitToDigisMapType& htd,
					   const SiPileUpSignals::signal_map_type& afterNoise,
					   unsigned int detID){
  digis.data.reserve(50); 
  digis.data.clear();
  link_coll.data.clear();  
  
  for ( DigitalMapType::const_iterator i=dm.begin(); i!=dm.end(); i++) {
    digis.data.push_back( SiStripDigi( (*i).first, (*i).second));
    ndigis++; 
  }
  
  // reworked to access the fraction of amplitude per simhit
  
  for ( HitToDigisMapType::const_iterator mi=htd.begin(); mi!=htd.end(); mi++) {
    
    if ((*((const_cast<DigitalMapType * >(&dm)))).find((*mi).first) != (*((const_cast<DigitalMapType * >(&dm)))).end() ){           
      // --- For each channel, sum up the signals from a simtrack
      
      map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
      for (vector < pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	     (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
	totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
      }
      
      //--- include the noise as well
      
      SiPileUpSignals::signal_map_type& temp1 = const_cast<SiPileUpSignals::signal_map_type&>(afterNoise);
      float totalAmplitude1 = temp1[(*mi).first];
      
      //--- digisimlink
      
      for (map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	   iter != totalAmplitudePerSimHit.end(); iter++){
	
	float threshold = 0.;
	if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {
	  float fraction = totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1;
	  
	  //Noise fluctuation could make fraction>1. Unphysical, set it by hand.
	  if(fraction >1.) fraction = 1.;
	  
	  link_coll.data.push_back(StripDigiSimLink( (*mi).first,   //channel
						     ((*iter).first)->trackId(), //simhit
						     fraction));    //fraction
	}
      }
    }
  }
}


