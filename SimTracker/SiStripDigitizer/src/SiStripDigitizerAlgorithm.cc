// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.
// Author:  A. Giammanco
// Creation Date:  Oct. 2, 2005   

#include <vector>
#include <iostream>

#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "Geometry/Surface/interface/BoundSurface.h"

using namespace std;

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet const& conf):conf_(conf){
  cout << "Creating a SiStripDigitizerAlgorithm." << endl;
  ndigis=0;
}
SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){
  cout << "Destroying a SiStripDigitizerAlgorithm." << endl;
}


//  Run the algorithm
//  ------------------
void SiStripDigitizerAlgorithm::run(const std::vector<PSimHit*> &input,
				      StripDigiCollection &output, StripGeomDetUnit *det)
{
  /// Temporary solution: run() contains what was in the constructor of SiStripDigitizer in ORCA.

  std::cout << "SiStripDigitizerAlgorithm is running!" << endl;

  NumberOfSegments = 20; // Default number of track segment divisions
  ClusterWidth = 3.;     // Charge integration spread on the collection plane
  Sigma0 = 0.0007;           // Charge diffusion constant 
  Dist300 = 0.0300;          //   normalized to 300micron Silicon
  theElectronPerADC=conf_.getParameter<double>("ElectronPerAdc");
  theThreshold=conf_.getParameter<double>("AdcThreshold");
  ENC=conf_.getParameter<double>("EquivalentNoiseCharge300um");
  noNoise=conf_.getParameter<bool>("NoNoise");
  peakMode=conf_.getParameter<bool>("APVpeakmode");
  if (peakMode) {
    tofCut=100;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      cout<<"APVs running in peak mode (poor time resolution)"<<endl;
    }
  } else {
    tofCut=50;
    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
      cout<<"APVs running in deconvolution mode (good time resolution)"<<endl;
    }
  };
  //  numStripsMax=conf_.getParameter<int>("NumStripsMax");
  //  cout<<theElectronPerADC<<" elettroni"<<endl;
 
  topol=&det->specificTopology(); // cache topology
  numStrips = topol->nstrips();  // det module number of strips
   
  //thickness:
  BoundSurface& p = (dynamic_cast<StripGeomDetUnit*>((det)))->surface();
  moduleThickness = p.bounds().thickness();
  float noiseRMS = ENC*moduleThickness/(0.03);
  float min_adc = theThreshold*noiseRMS/theElectronPerADC;
  std::cout << "module thickness: " << moduleThickness << ", noiseRMS: " << noiseRMS << ", min_adc: " << min_adc << endl;

  theSiNoiseAdder = new SiGaussianTailNoiseAdder(numStrips,noiseRMS,theThreshold);
  theSiZeroSuppress = new SiTrivialZeroSuppress(conf_,noiseRMS/theElectronPerADC); 
  theSiHitDigitizer = new SiHitDigitizer(conf_,det);
  theSiPileUpSignals = new SiPileUpSignals();
  theSiDigitalConverter = new SiTrivialDigitalConverter(theElectronPerADC);

  //};


  //vector<StripDigi> SiStripDigitizerAlgorithm::digitize(const std::vector<PSimHit*> &input,StripDigiCollection &output,StripGeomDetUnit *det){

  theSiPileUpSignals->reset();
  unsigned int detID = 0; // AG
  bool first = true; // AG

  //
  // First: loop on the SimHits
  //
  vector<PSimHit*>::const_iterator simHitIter = input.begin();
  vector<PSimHit*>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    if (ndigis%100 == 0) cout << "# digis: " << ndigis << endl;
    ++ndigis;
    //pointer to the simhit
    const PSimHit *ihit = *simHitIter;
    // detID (AG)
    if ( first ) {
      detID = ihit->detUnitId();
      first = false;
    }
    //
    // Compute the different charges;
    //
    if ( abs(ihit->tof()) < tofCut && ihit->energyLoss()>0) {
      SiHitDigitizer::hit_map_type _temp = theSiHitDigitizer->processHit(*ihit,*det);
      theSiPileUpSignals->add(_temp,*ihit);
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
  push_digis(output,theSiZeroSuppress->zeroSuppress(theSiDigitalConverter->convert(afterNoise)),
	       theLink,afterNoise,detID);



}

 void SiStripDigitizerAlgorithm::push_digis(StripDigiCollection &output,
					    const DigitalMapType& dm,
					    const HitToDigisMapType& htd,
					    const SiPileUpSignals::signal_map_type& afterNoise,
					    unsigned int detID){

   static vector<StripDigi> digis; 
   digis.reserve(50); 
   digis.clear();
   for ( DigitalMapType::const_iterator i=dm.begin(); 
	 i!=dm.end(); i++) {
     digis.push_back( StripDigi( (*i).first, (*i).second));
   }

   //det.specificReadout().addDigis( digis); // ???

   //
   // reworked to access the fraction of amplitude per simhit
   //
   for ( HitToDigisMapType::const_iterator mi=htd.begin();
	 mi!=htd.end(); mi++) {
     if ((*((const_cast<DigitalMapType * >(&dm))))[(*mi).first] != 0){
       //
       // For each channel, sum up the signals from a simtrack
       //
       map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
       for (vector < pair < const PSimHit*, Amplitude > >::const_iterator simul = 
              (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
         totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
       }
       //
       // no, I do it in another way: I take the total amplitude (included noise!)
       //      
       SiPileUpSignals::signal_map_type& temp1 = const_cast<SiPileUpSignals::signal_map_type&>(afterNoise);
       float totalAmplitude1 = temp1[(*mi).first];
       //
       // I push the links
       //
       for (map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	    iter != totalAmplitudePerSimHit.end(); iter++){
	 //
	 // Save only if the fraction if greater than something
	 //
	 float threshold = 0.;
	 if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {
	   // add link...
	   // in ORCA it was done this way:
	   /* det.simDet()->addLink((*mi).first,
	      (*iter).first->packedTrackId(),
	      totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1); */
	 }

       }

     }

   }

   //Fill the stripidigicollection
   StripDigiCollection::Range outputRange;
   outputRange.first = digis.begin();
   outputRange.second = digis.end();
   output.put(outputRange,detID);
   digis.clear();

 }


