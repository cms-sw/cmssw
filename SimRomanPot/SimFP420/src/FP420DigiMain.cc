///////////////////////////////////////////////////////////////////////////////
// File: FP420DigiMain.cc
// Date: 12.2006
// Description: FP420DigiMain for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"

#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/ChargeDividerFP420.h"
//#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



#include <vector>
#include <iostream>
using namespace std;

//#define CBOLTZ (1.38E-23)
//#define e_SI (1.6E-19)

//#define mydigidebug1





FP420DigiMain::FP420DigiMain(const edm::ParameterSet& conf):conf_(conf){
  std::cout << "Creating a FP420DigiMain" << std::endl;
  ndigis=0;
  edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("FP420DigiMain");
  verbosity    = m_Anal.getParameter<int>("Verbosity");
  theElectronPerADC = m_Anal.getParameter<double>("ElectronFP420PerAdc");
  theThreshold      = m_Anal.getParameter<double>("AdcFP420Threshold");
  noNoise           = m_Anal.getParameter<bool>("NoFP420Noise");
  
  
  if(verbosity>0) {
    std::cout << "FP420DigiMain theElectronPerADC=" << theElectronPerADC << " theThreshold=" << theThreshold << " noNoise=" << noNoise << std::endl;
  }
  
  Thick300 = 0.300;       // = 0.300 mm  normalized to 300micron Silicon
  ENC= 2160.;             //          EquivalentNoiseCharge300um = 2160. + other sources of noise
  ldriftX = 0.050;        // in mm
  ldriftY = 0.050;        // in mm
  pitchY= 0.050;          // in mm
  pitchX= 0.050;          // in mm
  moduleThickness = 0.250; // mm
  numStripsX = 401;        // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
  numStripsY = 201;        // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
  
  if(verbosity>0) {
    std::cout << "FP420DigiMain moduleThickness=" << moduleThickness << std::endl;
  }
  
  
  
  theFP420NumberingScheme = new FP420NumberingScheme();
  
  float noiseRMS = ENC*moduleThickness/Thick300;

  theZSuppressFP420 = new ZeroSuppressFP420(conf_,noiseRMS/theElectronPerADC); 
  thePileUpFP420 = new PileUpFP420();
  theDConverterFP420 = new DigiConverterFP420(theElectronPerADC);
  
  if(verbosity>0) {
    std::cout << "FP420DigiMain end of constructor" << std::endl;
  }
}
FP420DigiMain::~FP420DigiMain(){
  if(verbosity>0) {
    std::cout << "Destroying a FP420DigiMain" << std::endl;
  }
  delete theGNoiseFP420;
  delete theZSuppressFP420;
  delete theHitDigitizerFP420;
  delete thePileUpFP420;
  delete theDConverterFP420;

}



//  Run the algorithm
//  ------------------

vector <HDigiFP420> FP420DigiMain::run(const std::vector<FP420G4Hit> &input,
				       G4ThreeVector bfield, unsigned int iu, int sScale)  {
  
  // unpack from iu:
  //  int  sScale = 20, zScale=2;
  int  zScale=2;
  int  sector = (iu-1)/sScale + 1 ;
  int  zmodule = (iu - (sector - 1)*sScale - 1) /zScale + 1 ;
  int  zside = iu - (sector - 1)*sScale - (zmodule - 1)*zScale ;
  
  // Y:
  if (zside ==1) {
    numStrips = numStripsY;  // Y plate number of strips:200*0.050=10mm --> 100*0.100=10mm
    pitch= pitchY;
    ldrift = ldriftX; // because drift is in local coordinates which 90 degree rotated ( for correct timeNormalization & noiseRMS calculations)
  }
  // X:
  if (zside ==2) {
    numStrips = numStripsX;  // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
    pitch= pitchX;
    ldrift = ldriftY; // because drift is in local coordinates which 90 degree rotated ( for correct timeNormalization & noiseRMS calculations)
  }
  
  float noiseRMS = ENC*moduleThickness/Thick300;
  
  
  theHitDigitizerFP420 = new HitDigitizerFP420(moduleThickness,ldrift,ldriftY,ldriftX);
  theGNoiseFP420 = new GaussNoiseFP420(numStrips,noiseRMS,theThreshold);

  
  
  
  thePileUpFP420->reset();
  //  unsigned int detID = det->globalId().rawId();
  //  unsigned int detID = 1;
  bool first = true; // AG
  
  // main loop (AZ) 
  //
  // First: loop on the SimHits
  //
  vector<FP420G4Hit>::const_iterator simHitIter = input.begin();
  vector<FP420G4Hit>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    
    const FP420G4Hit ihit = *simHitIter;
    // detID (AG)
    if ( first ) {
      //       detID = ihit.detUnitId();    // AZ
      first = false;
    }
    // main part here (AZ):
    double  losenergy = ihit.getEnergyLoss();
    //      float   tof = ihit.getTof();
#ifdef mydigidebug1
    unsigned int unitID = ihit.getUnitID();
    //      int det, zside, sector, zmodule;
    //      FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
    //        int  sScale = 20;
    // intindex is a continues numbering of FP420
    //	  int zScale=2;  unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside; 
    // int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
    std::cout << " *******FP420DigiMain:                           intindex= " << iu << std::endl;
    std::cout << " tof= " << tof << "  EnergyLoss= " << losenergy << std::endl;
    std::cout << " EntryLocalP= " << ihit.getEntryLocalP() << std::endl;
    std::cout << " ExitLocalP= " << ihit.getExitLocalP() << std::endl;
    std::cout << " unitID= " << unitID << "  zside= " << zside << std::endl;
    std::cout << " sector= " << sector << "  zmodule= " << zmodule << std::endl;
    std::cout << "  bfield= " << bfield << std::endl;
#endif
    
    //        if ( abs(tof) < tofCut && losenergy>0) {
    if ( losenergy>0) {
      
      //   zside = 1 - Y strips;   =2 - X strips;
      //	  HitDigitizerFP420::hit_map_type _temp = theHitDigitizerFP420->processHit(ihit,bfield,zside,numStrips,pitch);
      HitDigitizerFP420::hit_map_type _temp = theHitDigitizerFP420->processHit(ihit,bfield,zside,numStrips,pitch,moduleThickness); 
      
      
      
#ifdef mydigidebug1
      std::cout << " *******FP420DigiMain: start:PileUpFP420->add" << std::endl;
#endif
      thePileUpFP420->add(_temp,ihit);
      
    }// if
    else {
      std::cout << " *******FP420DigiMain: ERROR???  losenergy =  " <<  losenergy  << std::endl;
    }
    // main part end (AZ) 
  } //for
  // main loop end (AZ) 
  
#ifdef mydigidebug1
  std::cout << "          " << std::endl;
  std::cout << " *******FP420DigiMain: END of LOOP on HITs !!!!!!!" << std::endl;
  std::cout << "          " << std::endl;
  std::cout << "          " << std::endl;
  std::cout << " *******FP420DigiMain: start:dumpSignal - return theMap" << std::endl;
#endif
  PileUpFP420::signal_map_type theSignal = thePileUpFP420->dumpSignal();
#ifdef mydigidebug1
  std::cout << " *******FP420DigiMain: start:dumpLink - return theMapLink" << std::endl;
#endif
  PileUpFP420::HitToDigisMapType theLink = thePileUpFP420->dumpLink();  
  
  
#ifdef mydigidebug1
  std::cout << " *******FP420DigiMain: start:afterNoise" << std::endl;
#endif
  PileUpFP420::signal_map_type afterNoise;
  
  
  
  
  if (noNoise) {
#ifdef mydigidebug1
    std::cout << " *******FP420DigiMain: start:IFnoNoise    theSignal" << std::endl;
#endif
    afterNoise = theSignal;
  } else {
#ifdef mydigidebug1
    std::cout << " *******FP420DigiMain: start:IFnoNoiseelse    addNoise" << std::endl;
#endif
    afterNoise = theGNoiseFP420->addNoise(theSignal);
  }
  
  
  
  
  digis.clear();
  
#ifdef mydigidebug1
  std::cout << " *******FP420DigiMain: start:push_digis zside = " << zside << std::endl;
#endif
  
  
  //                                                                                                                !!!!!
  push_digis(theZSuppressFP420->zeroSuppress(theDConverterFP420->convert(afterNoise)),
	     theLink,afterNoise);
  
  
  
  return digis; // to HDigiFP420
}

void FP420DigiMain::push_digis(const DigitalMapType& dm,
			       const HitToDigisMapType& htd,
			       const PileUpFP420::signal_map_type& afterNoise
			       ){
  
#ifdef mydigidebug1
  std::cout << " ****FP420DigiMain: push_digis start: do for loop dm.size()=" <<  dm.size() << std::endl;
#endif
  
  
  digis.reserve(50); 
  digis.clear();
  //   link_coll.clear();
  for ( DigitalMapType::const_iterator i=dm.begin(); i!=dm.end(); i++) {
    
    // push to digis the content of first and second words of HDigiFP420 vector for every strip pointer (*i)
    digis.push_back( HDigiFP420( (*i).first, (*i).second));
    ndigis++; 
    // very useful check:
#ifdef mydigidebug1
    std::cout << " ****FP420DigiMain:push_digis:  ndigis = " << ndigis << std::endl;
    std::cout << "push_digis: strip  = " << (*i).first << "  adc = " << (*i).second << std::endl;
    
#endif
  }
  
  ////////////////////////////det.specificReadout().addDigis( digis); // 
         
  //
  // reworked to access the fraction of amplitude per simhit FP420G4Hit
  //
  for ( HitToDigisMapType::const_iterator mi=htd.begin(); mi!=htd.end(); mi++) {
  #ifdef mydigidebug1
  std::cout << " ****push_digis:first for:  (*mi).first = " << (*mi).first << std::endl;
  std::cout << " if condition   = " << (*((const_cast<DigitalMapType * >(&dm))))[(*mi).first] << std::endl;
  #endif
  //    if ((*((const_cast<DigitalMapType * >(&dm))))[(*mi).first] != 0){
  if ((*((const_cast<DigitalMapType * >(&dm)))).find((*mi).first) != (*((const_cast<DigitalMapType * >(&dm)))).end() ){
  //
  // For each channel, sum up the signals from a simtrack
  //
  map<const FP420G4Hit *, Amplitude> totalAmplitudePerSimHit;
  for (vector < pair < const FP420G4Hit*, Amplitude > >::const_iterator simul = 
  (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
  #ifdef mydigidebug1
  std::cout << " ****push_digis:inside last for: (*simul).second= " << (*simul).second << std::endl;
  #endif
  totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
  } // for
  } // if
  } // for


  /////////////////////////////////////////////////////////////////////////////////////////////
  /*     
  // reworked to access the fraction of amplitude per simhit
  
  for ( HitToDigisMapType::const_iterator mi=htd.begin(); mi!=htd.end(); mi++) {
  
  if ((*((const_cast<DigitalMapType * >(&dm)))).find((*mi).first) != (*((const_cast<DigitalMapType * >(&dm)))).end() ){
  // --- For each channel, sum up the signals from a simtrack
  
  map<const FP420G4Hit *, Amplitude> totalAmplitudePerSimHit;
  for (vector < pair < const FP420G4Hit*, Amplitude > >::const_iterator simul =
  (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
  totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
  }

  //--- include the noise as well
  
  PileUpFP420::signal_map_type& temp1 = const_cast<PileUpFP420::signal_map_type&>(afterNoise);
  float totalAmplitude1 = temp1[(*mi).first];
  
  //--- digisimlink
  
  for (map<const FP420G4Hit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin();
  iter != totalAmplitudePerSimHit.end(); iter++){
  
  float threshold = 0.;
  if (totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1 >= threshold) {
  float fraction = totalAmplitudePerSimHit[(*iter).first]/totalAmplitude1;
  
  //Noise fluctuation could make fraction>1. Unphysical, set it by hand.
  if(fraction >1.) fraction = 1.;
  
  link_coll.push_back(StripDigiSimLink( (*mi).first,   //channel
  ((*iter).first)->trackId(), //simhit
  fraction));    //fraction
  }//if
  }//for
  }//if
  }//for
  
  */
  
  
  
}
