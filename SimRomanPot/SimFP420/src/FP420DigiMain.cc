///////////////////////////////////////////////////////////////////////////////
// File: FP420DigiMain.cc
// Date: 12.2006
// Description: FP420DigiMain for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"

#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/ChargeDividerFP420.h"

//#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
//#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
//#include "DataFormats/Common/interface/DetSetVector.h"
//#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"




#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <gsl/gsl_sf_erf.h>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Here zside = xytype - type of plate

#include <vector>
#include <iostream>
using namespace std;

//#define CBOLTZ (1.38E-23)
//#define e_SI (1.6E-19)



FP420DigiMain::FP420DigiMain(const edm::ParameterSet& conf):conf_(conf){
  std::cout << "Creating a FP420DigiMain" << std::endl;
  ndigis=0;
  verbosity        = conf_.getUntrackedParameter<int>("VerbosityLevel");
  theElectronPerADC= conf_.getParameter<double>("ElectronFP420PerAdc");
  theThreshold     = conf_.getParameter<double>("AdcFP420Threshold");
  noNoise          = conf_.getParameter<bool>("NoFP420Noise");
  addNoisyPixels   = conf_.getParameter<bool>("AddNoisyPixels");
  thez420          = conf_.getParameter<double>("z420");
  thezD2           = conf_.getParameter<double>("zD2");
  thezD3           = conf_.getParameter<double>("zD3");
  theApplyTofCut   = conf_.getParameter<bool>("ApplyTofCut");
  tofCut           = conf_.getParameter<double>("LowtofCutAndTo200ns");
  xytype=2;
  
  if(verbosity>0) {
    std::cout << "theApplyTofCut=" << theApplyTofCut << " tofCut=" << tofCut << std::endl;
    std::cout << "FP420DigiMain theElectronPerADC=" << theElectronPerADC << " theThreshold=" << theThreshold << " noNoise=" << noNoise << std::endl;
  }
  // X (or Y)define type of sensor (zside=1 or 2 used to derive it: 1-Y, 2-X)
  // for every type there is normal pixel size=0.05 and Wide 0.400 mm
  Thick300 = 0.300;       // = 0.300 mm  normalized to 300micron Silicon
  //ENC= 2160.;             //          EquivalentNoiseCharge300um = 2160. + other sources of noise
  ENC= 960.;             //          EquivalentNoiseCharge300um = 2160. + other sources of noise
  
  ldriftX = 0.050;        // in mm(zside=1)
  ldriftY = 0.050;        // in mm(zside=2)
  moduleThickness = 0.250; // mm(zside=1)(zside=2)
  
  pitchY= 0.050;          // in mm(zside=1)
  pitchX= 0.050;          // in mm(zside=2)
  numStripsY = 201;        // Y plate number of strips:200*0.050=10mm (zside=1)
  numStripsX = 401;        // X plate number of strips:400*0.050=20mm (zside=2)
  
  pitchYW= 0.400;          // in mm(zside=1)
  pitchXW= 0.400;          // in mm(zside=2)
  numStripsYW = 51;        // Y plate number of W strips:50 *0.400=20mm (zside=1) - W have ortogonal projection
  numStripsXW = 26;        // X plate number of W strips:25 *0.400=10mm (zside=2) - W have ortogonal projection
  
  //  tofCut = 1350.;           // Cut on the particle TOF range  = 1380 - 1500
  elossCut = 0.00003;           // Cut on the particle TOF   = 100 or 50
  
  
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

vector <HDigiFP420> FP420DigiMain::run(const std::vector<PSimHit> &input,
				       G4ThreeVector bfield, unsigned int iu )  {
  //				       G4ThreeVector bfield, unsigned int iu, int sScale)  {
  
  //  // unpack from iu:
  //  //  int  sScale = 20, zScale=2;
  //  int  zScale=2;
  //  int  sector = (iu-1)/sScale + 1 ;
  //  int  zmodule = (iu - (sector - 1)*sScale - 1) /zScale + 1 ;
  //  int  zside = iu - (sector - 1)*sScale - (zmodule - 1)*zScale ;
  //  if(verbosity>10) {
  //    std::cout << "FP420DigiMain xytype=" << xytype << " zside=" << zside << " zmodule=" << zmodule << " sector=" << sector << std::endl;
  //  }
  
  // Y:
  if (xytype ==1) {
    numStrips = numStripsY;  // Y plate number of strips:200*0.050=10mm --> 100*0.100=10mm
    pitch= pitchY;
    ldrift = ldriftX; // because drift is in local coordinates which 90 degree rotated ( for correct timeNormalization & noiseRMS calculations)
    numStripsW = numStripsYW;  // Y plate number of strips:200*0.050=10mm --> 100*0.100=10mm
    pitchW= pitchYW;
  }
  // X:
  if (xytype ==2) {
    numStrips = numStripsX;  // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
    pitch= pitchX;
    ldrift = ldriftY; // because drift is in local coordinates which 90 degree rotated ( for correct timeNormalization & noiseRMS calculations)
    numStripsW = numStripsXW;  // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
    pitchW= pitchXW;
  }
  
  float noiseRMS = ENC*moduleThickness/Thick300;
  
  
  theHitDigitizerFP420 = new HitDigitizerFP420(moduleThickness,ldrift,ldriftY,ldriftX,thez420,thezD2,thezD3);
  int numPixels = numStrips*numStripsW;
  theGNoiseFP420 = new GaussNoiseFP420(numPixels,noiseRMS,theThreshold,addNoisyPixels);
  //  theGNoiseFP420 = new GaussNoiseFP420(numStrips,noiseRMS,theThreshold,addNoisyPixels);
  
  
  
  
  thePileUpFP420->reset();
  //  unsigned int detID = det->globalId().rawId();
  //  unsigned int detID = 1;
  bool first = true; // AG
  
  // main loop (AZ) 
  //
  // First: loop on the SimHits
  //
  vector<PSimHit>::const_iterator simHitIter = input.begin();
  vector<PSimHit>::const_iterator simHitIterEnd = input.end();
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    
    const PSimHit ihit = *simHitIter;
    // detID (AG)
    if ( first ) {
      //       detID = ihit.detUnitId();    // AZ
      first = false;
    }
    // main part here (AZ):
    double  losenergy = ihit.energyLoss();
    float   tof = ihit.tof();
    //ouble  losenergy = ihit.getEnergyLoss();
    //float   tof = ihit.getTof();
    if(verbosity>0) {
      unsigned int unitID = ihit.detUnitId();
      //    //      int det, zside, sector, zmodule;
      //    //      FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
      //    //        int  sScale = 20;
      //    // intindex is a continues numbering of FP420
      //    //	  int zScale=2;  unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside; 
      //    // int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
      std::cout << " *******FP420DigiMain:                           intindex= " << iu << std::endl;
      std::cout << " tof= " << tof << "  EnergyLoss= " << losenergy  << "  pabs= " <<  ihit.pabs() << std::endl;
      std::cout << " EntryLocalP x= " << ihit.entryPoint().x() << " EntryLocalP y= " << ihit.entryPoint().y() << std::endl;
      std::cout << " ExitLocalP x= " << ihit.exitPoint().x() << " ExitLocalP y= " << ihit.exitPoint().y() << std::endl;
      std::cout << " EntryLocalP z= " << ihit.entryPoint().z() << " ExitLocalP z= " << ihit.exitPoint().z() << std::endl;
      std::cout << " unitID= " << unitID << "  xytype= " << xytype << " particleType= " << ihit.particleType() << " trackId= " << ihit.trackId() << std::endl;
      //    std::cout << " sector= " << sector << "  zmodule= " << zmodule << std::endl;
      std::cout << "  bfield= " << bfield << std::endl;
    }

  if(verbosity>0) {
      std::cout << " *******FP420DigiMain:                           theApplyTofCut= " << theApplyTofCut << std::endl;
      std::cout << " tof= " << tof << "  EnergyLoss= " << losenergy  << "  pabs= " <<  ihit.pabs() << std::endl;
      std::cout << " particleType= " << ihit.particleType() << std::endl;
  }
    if ( ( !(theApplyTofCut)  ||  (theApplyTofCut &&   tofCut < abs(tof) < (tofCut+200.)) ) && losenergy > elossCut) {
      //    if ( abs(tof) < tofCut && losenergy > elossCut) {
      // if ( losenergy>0) {
      if(verbosity>0) std::cout << " inside tof: OK " << std::endl;
      
      //   zside = 1 - Y strips;   =2 - X strips;
      //	  HitDigitizerFP420::hit_map_type _temp = theHitDigitizerFP420->processHit(ihit,bfield,zside,numStrips,pitch);
      HitDigitizerFP420::hit_map_type _temp = theHitDigitizerFP420->processHit(ihit,bfield,xytype,numStrips,pitch,numStripsW,pitchW,moduleThickness); 
      
      
      
      thePileUpFP420->add(_temp,ihit);
      
    }// if
    else {
      //    std::cout << " *******FP420DigiMain: ERROR???  losenergy =  " <<  losenergy  << std::endl;
    }
    // main part end (AZ) 
  } //for
  // main loop end (AZ) 
  
  PileUpFP420::signal_map_type theSignal = thePileUpFP420->dumpSignal();
  PileUpFP420::HitToDigisMapType theLink = thePileUpFP420->dumpLink();  
  
  
  PileUpFP420::signal_map_type afterNoise;
  
  if (noNoise) {
    afterNoise = theSignal;
  } else {
    afterNoise = theGNoiseFP420->addNoise(theSignal);
    //    add_noise();
  }
  
  //  if((pixelInefficiency>0) && (_signal.size()>0)) 
  //  pixel_inefficiency(); // Kill some pixels
  
  
  
  digis.clear();
  
  
  
  //                                                                                                                !!!!!
  push_digis(theZSuppressFP420->zeroSuppress(theDConverterFP420->convert(afterNoise)),
	     theLink,afterNoise);
  
  
  
  return digis; // to HDigiFP420
}





void FP420DigiMain::push_digis(const DigitalMapType& dm,
			       const HitToDigisMapType& htd,
			       const PileUpFP420::signal_map_type& afterNoise
			       )        {
  //  
  if(verbosity>0) {
    std::cout << " ****FP420DigiMain: push_digis start: do for loop dm.size()=" <<  dm.size() << std::endl;
  }
  
  
  digis.reserve(50); 
  digis.clear();
  //   link_coll.clear();
  for ( DigitalMapType::const_iterator i=dm.begin(); i!=dm.end(); i++) {
    
    // Load digis
    // push to digis the content of first and second words of HDigiFP420 vector for every strip pointer (*i)
    digis.push_back( HDigiFP420( (*i).first, (*i).second));
    ndigis++; 
    // very useful check:
    if(verbosity>0) {
      std::cout << " ****FP420DigiMain:push_digis:  ndigis = " << ndigis << std::endl;
      std::cout << "push_digis: strip  = " << (*i).first << "  adc = " << (*i).second << std::endl;
    }    
    
  }
  
  ////////////////////////////det.specificReadout().addDigis( digis); // 
  /*       
  //
  // reworked to access the fraction of amplitude per simhit PSimHit
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
  map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
  for (vector < pair < const PSimHit*, Amplitude > >::const_iterator simul = 
  (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
  #ifdef mydigidebug1
  std::cout << " ****push_digis:inside last for: (*simul).second= " << (*simul).second << std::endl;
  #endif
  totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
  } // for
  }  // if
  } // for
  */
  
  /////////////////////////////////////////////////////////////////////////////////////////////
    
    // reworked to access the fraction of amplitude per simhit
    
    for ( HitToDigisMapType::const_iterator mi=htd.begin(); mi!=htd.end(); mi++) {
      //      
      if ((*((const_cast<DigitalMapType * >(&dm)))).find((*mi).first) != (*((const_cast<DigitalMapType * >(&dm)))).end() ){
	// --- For each channel, sum up the signals from a simtrack
	//
	map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
	for (vector < pair < const PSimHit*, Amplitude > >::const_iterator simul =
	       (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
	  totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
	}
	/*	
	//--- include the noise as well
	
	PileUpFP420::signal_map_type& temp1 = const_cast<PileUpFP420::signal_map_type&>(afterNoise);
	float totalAmplitude1 = temp1[(*mi).first];
	
	//--- digisimlink
	for (map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin();
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
	*/
	//
      }//if
    }//for
    //
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
      //      
      //      
      //      
      }
