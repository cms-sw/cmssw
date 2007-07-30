#include "SimTracker/SiStripDigitizer/interface/SiTrivialZeroSuppress.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "Utilities/Notification/interface/TimingReport.h" // ???

//Author: Matthew Pearson, March 2003, ported to CMSSW by Andrea Giammanco, November 2005

//Constructor
SiTrivialZeroSuppress::SiTrivialZeroSuppress(const edm::ParameterSet& conf, float noise) : conf_(conf), theNumFEDalgos(4)
{
  noiseInAdc=noise;
  initParams(conf_);
}

//Read parameters needed for the zero suppression algorithm 
//(which is in SiTrivialZeroSuppress::trkFEDclusterizer)
void SiTrivialZeroSuppress::initParams(edm::ParameterSet const& conf_)
{
  algoConf=conf_.getParameter<int>("FedAlgorithm");
  lowthreshConf=conf_.getParameter<double>("FedLowThreshold");
  highthreshConf=conf_.getParameter<double>("FedHighThreshold");

  theFEDalgorithm  = algoConf;
  theFEDlowThresh  = lowthreshConf * noiseInAdc;
  theFEDhighThresh = highthreshConf * noiseInAdc;

  theNextFEDlowThresh  = theFEDlowThresh;
  theNext2FEDlowThresh = theFEDlowThresh;
  thePrevFEDlowThresh  = theFEDlowThresh;
  thePrev2FEDlowThresh = theFEDlowThresh;
  theNeighFEDlowThresh = theFEDlowThresh;
  
  theNextFEDhighThresh  = theFEDhighThresh;
  thePrevFEDhighThresh  = theFEDhighThresh;
  theNeighFEDhighThresh = theFEDhighThresh;
  
  //Check zero suppress algorithm
  if (theFEDalgorithm < 1 || theFEDalgorithm > theNumFEDalgos) {
    edm::LogError("StripDigiInfo")<<"SiTrivialZeroSuppress FATAL ERROR: Unknown zero suppress algorithm "<<theFEDalgorithm;
    exit(1);
  }
  
  //Check thresholds
  if (theFEDlowThresh > theFEDhighThresh) {
    edm::LogError("StripDigiInfo")<<"SiTrivialZeroSuppress FATAL ERROR: Low threshold exceeds high threshold: "<<theFEDlowThresh<<" > "<<theFEDhighThresh;
    exit(2);
  }
}

//Zero suppress method, which called the SiTrivialZeroSuppress::trkFEDclusterizer
SiZeroSuppress::DigitalMapType SiTrivialZeroSuppress::zeroSuppress(const DigitalMapType& notZeroSuppressedMap)
{
  //  const string s1("SiTrivialZeroSuppress::zeroSuppress");
  //  TimeMe time_me(s1);
  
  return trkFEDclusterizer(notZeroSuppressedMap); 
}

//This performs the zero suppression
SiZeroSuppress::DigitalMapType SiTrivialZeroSuppress::trkFEDclusterizer(const DigitalMapType &in) 
{
  const std::string s2("SiTrivialZeroSuppress::trkFEDclusterizer1");
  //  TimeMe time_me(s2); // ???
  
  DigitalMapType selectedSignal;
  register DigitalMapType::const_iterator i, iPrev, iNext, iPrev2, iNext2;
  
  for (i = in.begin(); i != in.end(); i++) {
  
    //Find adc values for neighbouring strips
    int strip = i->first;
    int adc   = i->second;
    iPrev  = in.find(strip - 1);
    iNext  = in.find(strip + 1);
    //Set values for strips just outside APV or module to infinity.
    //This is to avoid losing strips at the edges, 
    //which otherwise would pass cuts if strips were next to each other.
    int adcPrev  = -9999;
    int adcNext  = -9999;
    int adcPrev2 = -9999;
    int adcNext2 = -9999;

    if ( ((strip)%128) == 127){
      adcNext = 0;
      theNextFEDlowThresh  = 9999;
      theNextFEDhighThresh = 9999;
    }
    if ( ((strip)%128) == 0){
      adcPrev = 0;
      thePrevFEDlowThresh  = 9999;
      thePrevFEDhighThresh = 9999; 
    }
    //Otherwise if strip was found then find it's ADC count.
    if ( iPrev  != in.end() ) adcPrev  = iPrev->second;
    if ( iNext  != in.end() ) adcNext  = iNext->second;

    if ( adcNext <= adcPrev){
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh  = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh  = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }
    
    //Find adc values for next neighbouring strips
    iPrev2 = in.find(strip - 2); 
    iNext2 = in.find(strip + 2);
    //See above 
    if ( ((strip)%128) == 126){ 
      adcNext2 = 0;
      theNext2FEDlowThresh  = 9999;
    }
    if ( ((strip)%128) == 1){
      adcPrev2 = 0; 
      thePrev2FEDlowThresh  = 99999;
    }
    if ( iPrev2 != in.end() ) adcPrev2 = iPrev2->second; 
    if ( iNext2 != in.end() ) adcNext2 = iNext2->second; 
 
    //    cout << " strip " << strip << " adc " << adc << " adcPrev " << adcPrev
    //   << " adcNext " << adcNext << " adcMaxNeigh " << adcMaxNeigh << endl;
    //cout << "  theFEDlowThresh " <<  theFEDlowThresh << " theFEDhighThresh " << theFEDhighThresh << endl;
 
    // Decide if this strip should be accepted.
    bool accept = false;
    switch (theFEDalgorithm) {
      
    case 1:
      accept = (adc >= theFEDlowThresh);
      break;
    case 2:
      accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh &&
					    adcMaxNeigh >= theNeighFEDlowThresh));
      break;
    case 3:
      accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh &&
					    adcMaxNeigh >= theNeighFEDhighThresh));
      break;
    case 4:
      accept = (
		(adc >= theFEDhighThresh)            //Test for adc>highThresh (same as algorithm 2)
		||
		(
		 (adc >= theFEDlowThresh)            //Test for adc>lowThresh, with neighbour adc>lowThresh (same as algorithm 2)
		 &&
		 (adcMaxNeigh >= theNeighFEDlowThresh)
		 ) 
		||
		(
		 (adc < theFEDlowThresh)        //Test for adc<lowThresh
		 &&     
		 (
		  (
		   (adcPrev  >= thePrevFEDhighThresh)    //with both neighbours>highThresh
		   &&
		   (adcNext  >= theNextFEDhighThresh)
		   ) 
		  ||
		  (
		   (adcPrev  >= thePrevFEDhighThresh)    //OR with previous neighbour>highThresh and
		   &&
		   (adcNext  >= theNextFEDlowThresh)     //both the next neighbours>lowThresh
		   &&
		   (adcNext2 >= theNext2FEDlowThresh)
		   )  
		  ||
		  (
		   (adcNext  >= theNextFEDhighThresh)    //OR with next neighbour>highThresh and
		   &&
		   (adcPrev  >= thePrevFEDlowThresh)     //both the previous neighbours>lowThresh
		   &&
		   (adcPrev2 >= thePrev2FEDlowThresh)
		   )  
		  ||
		  (
		   (adcNext  >= theNextFEDlowThresh)     //OR with both next neighbours>lowThresh and
		   &&
		   (adcNext2 >= theNext2FEDlowThresh)   //both the previous neighbours>lowThresh
		   &&
		   (adcPrev  >= thePrevFEDlowThresh)  
		   &&
		   (adcPrev2 >= thePrev2FEDlowThresh)
		   )
		  )
		 )
		);
      break;
    }

    if (accept) {   
      selectedSignal[strip] = adc;
      //    cout << " selected strip: " << strip << "  adc: " << adc << endl;
    }  
  }
  return selectedSignal;
}
