//class SiPixelDigitizerAlgorithm SimTracker/SiPixelDigitizer/src/SiPixelDigitizerAlgoithm.cc 

// Original Author Danek Kotlinski
// Ported in CMSSW by  Michele Pioppi-INFN perugia
// Added DB capabilities by F.Blekman, Cornell University
//         Created:  Mon Sep 26 11:08:32 CEST 2005
// Add tof, change AddNoise to tracked. 4/06
// Change drift direction. 6/06 d.k.
// Add the statuis (non-rate dependent) inefficiency.
//     -1 - no ineffciency
//      0 - static inefficency only
//    1,2 - low-lumi rate dependent inefficency added
//     10 - high-lumi inefficiency added
// Adopt the correct drift sign convetion from Morris Swartz. d.k. 8/06
// Add more complex misscalinbration, change kev/e to 3.61, diff=3.7,d.k.9/06
// Add the readout channel electronic noise. d.k. 3/07
// Lower the pixel noise from 500 to 175elec.
// Change the input threshold from noise units to electrons.
// Lower the amount of static dead pixels from 0.01 to 0.001.
// Modify to the new random number services. d.k. 5/07
// Protect against sigma=0 (delta tracks on the surface). d.k.5/07
// Change the TOF cut to lower and upper limit. d.k. 7/07
//
// July 2008: Split Lorentz Angle configuration in BPix/FPix (V. Cuplov)
// tanLorentzAngleperTesla_FPix=0.0912 and tanLorentzAngleperTesla_BPix=0.106
// Sept. 2008: Disable Pixel modules which are declared dead in the configuration python file. (V. Cuplov)
// Oct. 2008: Accessing/Reading the Lorentz angle from the DataBase instead of the cfg file. (V. Cuplov)
// Accessing dead modules from the DB. Implementation done and tested on a test.db
// Do not use this option for now. The PixelQuality Objects are not in the official DB yet.
// Feb. 2009: Split Fpix and Bpix threshold and use official numbers (V. Cuplov)
// ThresholdInElectrons_FPix = 2870 and ThresholdInElectrons_BPix = 3700
// update the electron to VCAL conversion using: VCAL_electrons = VCAL * 65.5 - 414
// Feb. 2009: Threshold gaussian smearing (V. Cuplov)
// March, 2009: changed DB access to *SimRcd objects (to de-couple the DB objects from reco chain) (F. Blekman) 
// May, 2009: Pixel charge VCAL smearing. (V. Cuplov)
// November, 2009: new parameterization of the pixel response. (V. Cuplov)
// December, 2009: Fix issue with different compilers.
// October, 2010: Improvement: Removing single dead ROC (V. Cuplov)
// November, 2010: Bug fix in removing TBMB/A half-modules (V. Cuplov)

#include <vector>
#include <iostream>

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"

#include <gsl/gsl_sf_erf.h>
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

//#include "SimTracker/SiPixelDigitizer/interface/PixelIndices.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"
 
// Accessing dead pixel modules from the DB:
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

#define TP_DEBUG // protect all LogDebug with ifdef. Takes too much CPU

void SiPixelDigitizerAlgorithm::init(const edm::EventSetup& es){
  if(use_ineff_from_db_){// load gain calibration service fromdb...
    theSiPixelGainCalibrationService_= new SiPixelGainCalibrationOfflineSimService(conf_);
    theSiPixelGainCalibrationService_->setESObjects( es );
  }

  fillDeadModules(es); // gets the dead module from config file or DB.
  fillLorentzAngle(es); // gets the Lorentz angle from the config file or DB.
  fillMapandGeom(es);  //gets the map and geometry from the DB (to kill ROCs)  
}

//=========================================================================

void SiPixelDigitizerAlgorithm::fillDeadModules(const edm::EventSetup& es){
  if(!use_deadmodule_DB_){
    DeadModules = conf_.getParameter<Parameters>("DeadModules"); // get dead module from cfg file
  }
  else{  // Get dead module from DB record 
    // ESHandle was defined in the header file   edm::ESHandle<SiPixelQuality> SiPixelBadModule_;
    es.get<SiPixelQualityRcd>().get(SiPixelBadModule_);
  }
}

void SiPixelDigitizerAlgorithm::fillLorentzAngle(const edm::EventSetup& es){
  if(!use_LorentzAngle_DB_){
    // Get the Lorentz angle from the cfg file:
    tanLorentzAnglePerTesla_FPix=conf_.getParameter<double>("TanLorentzAnglePerTesla_FPix");
    tanLorentzAnglePerTesla_BPix=conf_.getParameter<double>("TanLorentzAnglePerTesla_BPix");
  } 
  else {
    // Get Lorentz angle from DB record 
    // ESHandle was defined in the header file edm::ESHandle<SiPixelLorentzAngle> SiPixelLorentzAngle_;
    es.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);
  }
}

// For ROC killing:
void SiPixelDigitizerAlgorithm::fillMapandGeom(const edm::EventSetup& es){
   es.get<SiPixelFedCablingMapRcd>().get(map_);
   es.get<TrackerDigiGeometryRecord>().get(geom_);
}

//=========================================================================

SiPixelDigitizerAlgorithm::SiPixelDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng) :
  conf_(conf) , fluctuate(0), theNoiser(0), pIndexConverter(0),
  use_ineff_from_db_(conf_.getParameter<bool>("useDB")),
  use_module_killing_(conf_.getParameter<bool>("killModules")), // boolean to kill or not modules
  use_deadmodule_DB_(conf_.getParameter<bool>("DeadModules_DB")), // boolean to access dead modules from DB
  use_LorentzAngle_DB_(conf_.getParameter<bool>("LorentzAngle_DB")), // boolean to access Lorentz angle from DB 
  theSiPixelGainCalibrationService_(0),rndEngine(eng)
{
  using std::cout;
  using std::endl;

  // Common pixel parameters
  // These are parameters which are not likely to be changed
  NumberOfSegments = 20; // Default number of track segment divisions
  ClusterWidth = 3.;     // Charge integration spread on the collection plane
  GeVperElectron = 3.61E-09; // 1 electron =3.61eV, 1keV=277e, mod 9/06 d.k.
  Sigma0 = 0.00037;           // Charge diffusion constant 7->3.7 
  Dist300 = 0.0300;          //   normalized to 300micron Silicon

  // Numbers of Layers
  NumberOfBarrelLayers = conf_.getParameter<int>("NumPixelBarrel");
  NumberOfEndcapDisks  = conf_.getParameter<int>("NumPixelEndcap");
  cout << "Number of Pixel Barrel Layer " << NumberOfBarrelLayers << endl;
  NumberOfTotLayers = NumberOfBarrelLayers + NumberOfEndcapDisks;

  alpha2Order = conf_.getParameter<bool>("Alpha2Order");   // switch on/off of E.B effect   

  // get external parameters:
  // ADC calibration 1adc count = 135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc]=2[adc/kev]
  // Be carefull, this parameter is also used in SiPixelDet.cc to 
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC=conf_.getParameter<double>("ElectronPerAdc");

  // ADC saturation value, 255=8bit adc.
  //theAdcFullScale=conf_.getUntrackedParameter<int>("AdcFullScale",255);
  theAdcFullScale=conf_.getParameter<int>("AdcFullScale");
  theAdcFullScaleStack=conf_.getParameter<int>("AdcFullScaleStack");
  theFirstStackLayer=conf_.getParameter<int>("FirstStackLayer");

  // Pixel threshold in units of noise:
  // thePixelThreshold=conf_.getParameter<double>("ThresholdInNoiseUnits");
  // Pixel threshold in electron units.
  theThresholdInE_FPix=conf_.getParameter<double>("ThresholdInElectrons_FPix");
  theThresholdInE_BPix=conf_.getParameter<double>("ThresholdInElectrons_BPix");
  //Carlotta" enables different threshold in layer 1 when we have smaller pixel pitch
  theThresholdInE_BPix_L1=conf_.getParameter<double>("ThresholdInElectrons_BPix_L1");

  // Add threshold gaussian smearing:
  addThresholdSmearing = conf_.getParameter<bool>("AddThresholdSmearing");
  theThresholdSmearing_FPix = conf_.getParameter<double>("ThresholdSmearing_FPix");
  theThresholdSmearing_BPix = conf_.getParameter<double>("ThresholdSmearing_BPix");

  // signal response new parameterization: split Fpix and BPix 
  FPix_p0 = conf_.getParameter<double>("FPix_SignalResponse_p0");
  FPix_p1 = conf_.getParameter<double>("FPix_SignalResponse_p1");
  FPix_p2 = conf_.getParameter<double>("FPix_SignalResponse_p2");
  FPix_p3 = conf_.getParameter<double>("FPix_SignalResponse_p3");

  BPix_p0 = conf_.getParameter<double>("BPix_SignalResponse_p0");
  BPix_p1 = conf_.getParameter<double>("BPix_SignalResponse_p1");
  BPix_p2 = conf_.getParameter<double>("BPix_SignalResponse_p2");
  BPix_p3 = conf_.getParameter<double>("BPix_SignalResponse_p3");

  // electrons to VCAL conversion needed in misscalibrate()
  electronsPerVCAL = conf_.getParameter<double>("ElectronsPerVcal");
  electronsPerVCAL_Offset = conf_.getParameter<double>("ElectronsPerVcal_Offset");

  // Add noise   
  addNoise=conf_.getParameter<bool>("AddNoise");

  // Smear the pixel charge with a gaussian which RMS is a function of the
  // pixel charge (Danek's study)
  addChargeVCALSmearing=conf_.getParameter<bool>("ChargeVCALSmearing");

  // Add noisy pixels 
  addNoisyPixels=conf_.getParameter<bool>("AddNoisyPixels");
  // Noise in electrons:
  // Pixel cell noise, relevant for generating noisy pixels 
  theNoiseInElectrons=conf_.getParameter<double>("NoiseInElectrons");
  // Fill readout noise, including all readout chain, relevant for smearing
  //theReadoutNoise=conf_.getUntrackedParameter<double>("ReadoutNoiseInElec",500.);
  theReadoutNoise=conf_.getParameter<double>("ReadoutNoiseInElec");

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  //theTofCut=conf_.getUntrackedParameter<double>("TofCut",12.5);
  theTofLowerCut=conf_.getParameter<double>("TofLowerCut");
  theTofUpperCut=conf_.getParameter<double>("TofUpperCut");
  
  // Fluctuate charge in track subsegments
  fluctuateCharge=conf_.getUntrackedParameter<bool>("FluctuateCharge",true);

  // delta cutoff in MeV, has to be same as in OSCAR=0.030/cmsim=1.0 MeV
  //tMax = 0.030; // In MeV.  
  //tMax =conf_.getUntrackedParameter<double>("DeltaProductionCut",0.030);  
  tMax =conf_.getParameter<double>("DeltaProductionCut");  
 
  // Control the pixel inefficiency
  thePixelLuminosity=conf_.getParameter<int>("AddPixelInefficiency");

  //--Hec: ADD HERE INNEFICIENCY INPUT PARARAMETERS [Sep-09]
  PixelEff     = conf_.getParameter<double>("PixelEff");       //--Hec: Pixel
  PixelColEff  = conf_.getParameter<double>("PixelColEff");    //--Hec: Column
  PixelChipEff = conf_.getParameter<double>("PixelChipEff");   //--Hec: Chip

  cout<<"My Eff Input: thePixelLuminosity= "<<thePixelLuminosity
  <<" PixelEff="<<PixelEff<<" PixelColEff="<<PixelColEff<<" PixelChipEff="<<PixelChipEff<<endl;   //--Hec:

  // Get the constants for the miss-calibration studies
  doMissCalibrate=conf_.getParameter<bool>("MissCalibrate"); // Enable miss-calibration
  theGainSmearing=conf_.getParameter<double>("GainSmearing"); // sigma of the gain smearing
  theOffsetSmearing=conf_.getParameter<double>("OffsetSmearing"); //sigma of the offset smearing

  //pixel inefficiency
  // the first 3 settings [0],[1],[2] are for the barrel pixels
  // the next  3 settings [3],[4],[5] are for the endcaps (undecided how)  

  if (thePixelLuminosity==-1) {  // No inefficiency, all 100% efficient
    pixelInefficiency=false;
    for (int i=0; i<NumberOfTotLayers;i++) {
      thePixelEfficiency[i]     = 1.;  // pixels = 100%
      thePixelColEfficiency[i]  = 1.;  // columns = 100%
      thePixelChipEfficiency[i] = 1.; // chips = 100%
      cout << "Calling No ineEfficiency: NumberOfTotLayers "<<i
           <<" Pixels: " <<thePixelEfficiency[i]
           <<" Columns: "<<thePixelColEfficiency[i]
           <<" Chips: "  <<thePixelChipEfficiency[i]<<endl; //--Hec:
    }
    
    // include only the static (non rate depedent) efficiency 
    // Usefull for very low rates (luminosity)
  } else if (thePixelLuminosity==0) { // static effciency
    pixelInefficiency=true;
    // Default efficiencies 
    for (int i=0; i<NumberOfTotLayers;i++) {
      if(i<NumberOfTotLayers) {  // For the barrel
	// Assume 1% inefficiency for single pixels, 
	// this is given by faulty bump-bonding and seus.  
	thePixelEfficiency[i]     = PixelEff; //--Hec: 1.-0.001;  // pixels = 99.9%
	// For columns make 0.1% default.
	thePixelColEfficiency[i]  = PixelColEff; //--Hec: 1.-0.001;  // columns = 99.9%
	// A flat 0.1% inefficiency due to lost rocs
	thePixelChipEfficiency[i] = PixelChipEff; //--Hec: 1.-0.001; // chips = 99.9%
	cout << "Calling Barrel Eff. NumberOfBarrelLayers: "<<i
	     <<" Pixels: " <<thePixelEfficiency[i]
	     <<" Columns: "<<thePixelColEfficiency[i]
	     <<" Chips: "  <<thePixelChipEfficiency[i]<<endl; //--Hec:
      } else { // For the endcaps
	// Assume 1% inefficiency for single pixels, 
	// this is given by faulty bump-bonding and seus.  
	thePixelEfficiency[i]     = PixelEff; //--Hec: 1.-0.001;  // pixels = 99.9%
	// For columns make 0.1% default.
	thePixelColEfficiency[i]  = PixelColEff; //--Hec: 1.-0.001;  // columns = 99.9%
	// A flat 0.1% inefficiency due to lost rocs
	thePixelChipEfficiency[i] = PixelChipEff; //--Hec: 1.-0.001; // chips = 99.9%
	cout << "Calling Barrel Eff. NumberOfBarrelLayers: "<<i
	     <<" Pixels: " <<thePixelEfficiency[i]
	     <<" Columns: "<<thePixelColEfficiency[i]
	     <<" Chips: "  <<thePixelChipEfficiency[i]<<endl; //--Hec:
      }
    }
    
    // Include also luminosity rate dependent inefficieny
  } else if (thePixelLuminosity>0) { // Include effciency
    cout << "Luminosity ineEfficiency [Need to be check]"<<endl; //--Hec:
    pixelInefficiency=true;
    // Default efficiencies 
    for (int i=0; i<NumberOfTotLayers;i++) {
      if(i<NumberOfBarrelLayers) { // For the barrel
	// Assume 1% inefficiency for single pixels, 
	// this is given by faulty bump-bonding and seus.  
	thePixelEfficiency[i]     = PixelEff; //--Hec: 1.-0.01;  // pixels = 99%
	// For columns make 1% default.
	thePixelColEfficiency[i]  = PixelColEff; //--Hec: 1.-0.01;  // columns = 99%
	// A flat 0.25% inefficiency due to lost data packets from TBM
	thePixelChipEfficiency[i] = PixelChipEff; //--Hec: 1.-0.0025; // chips = 99.75%
      } else { // For the endcaps
	// Assume 1% inefficiency for single pixels, 
	// this is given by faulty bump-bonding and seus.  
	thePixelEfficiency[i]     = PixelEff; //--Hec: 1.-0.01;  // pixels = 99%
	// For columns make 1% default.
	thePixelColEfficiency[i]  = PixelColEff; //--Hec: 1.-0.01;  // columns = 99%
	// A flat 0.25% inefficiency due to lost data packets from TBM
	thePixelChipEfficiency[i] = PixelChipEff; //--Hec: 1.-0.0025; // chips = 99.75%
      }
    }
   
    // Special cases ( High-lumi for 4cm layer) where the readout losses are higher
    if(thePixelLuminosity==10) { // For high luminosity, bar layer 1
      thePixelColEfficiency[0] = PixelColEff; //--Hec: 1.-0.034; // 3.4% for r=4 only
      thePixelEfficiency[0]    = PixelEff; //--Hec: 1.-0.015; // 1.5% for r=4
      thePixelChipEfficiency[0] = PixelChipEff; //--Hec: I added this variable here
    }
     if(thePixelLuminosity==20) { // For <PU>=50 phase 1 geometry
       thePixelColEfficiency[0] = 1.-0.0238; // i.e. 2.38% inefficiency for R30F12
       thePixelColEfficiency[1] = 1.-0.0046;
       thePixelColEfficiency[2] = 1.-0.0018;
       thePixelColEfficiency[3] = 1.-0.0008;
       thePixelColEfficiency[4] = 1.-0.0018;
     }
     if(thePixelLuminosity==30) { // For <PU>=100 phase 1 geometry
       thePixelColEfficiency[0] = 1.-0.0476; // for R30F12
       thePixelColEfficiency[1] = 1.-0.0093;
       thePixelColEfficiency[2] = 1.-0.0036;
       thePixelColEfficiency[3] = 1.-0.0017;
       thePixelColEfficiency[4] = 1.-0.0036;
     }
    
  } // end the pixel inefficiency part

  // Init the random number services  
  if(addNoise || thePixelLuminosity || fluctuateCharge || addThresholdSmearing ) {
    gaussDistribution_ = new CLHEP::RandGaussQ(rndEngine, 0., theReadoutNoise);
    gaussDistributionVCALNoise_ = new CLHEP::RandGaussQ(rndEngine, 0., 1.);
    flatDistribution_ = new CLHEP::RandFlat(rndEngine, 0., 1.);
    
    if(addNoise) { 
      theNoiser = new GaussianTailNoiseGenerator(rndEngine);
    }
    
    if(fluctuateCharge) {
      fluctuate = new SiG4UniversalFluctuation(rndEngine);
    }

    // Threshold smearing with gaussian distribution:
    if(addThresholdSmearing) {
      smearedThreshold_FPix_ = new CLHEP::RandGaussQ(rndEngine, theThresholdInE_FPix , theThresholdSmearing_FPix);
      smearedThreshold_BPix_ = new CLHEP::RandGaussQ(rndEngine, theThresholdInE_BPix , theThresholdSmearing_BPix);
      //Carlotta
      smearedThreshold_BPix_L1_ = new CLHEP::RandGaussQ(rndEngine, theThresholdInE_BPix_L1 , theThresholdSmearing_BPix);
    }
    } //end Init the random number services


  // Prepare for the analog amplitude miss-calibration
  if(doMissCalibrate) {
    LogDebug ("PixelDigitizer ") 
      << " miss-calibrate the pixel amplitude "; 
    
    const bool ReadCalParameters = false;
    if(ReadCalParameters) {   // Read the calibration files from file
      // read the calibration constants from a file (testing only)
      ifstream in_file;  // data file pointer
      char filename[80] = "phCalibrationFit_C0.dat";
      
      in_file.open(filename, ios::in ); // in C++
      if (in_file.bad()) {
	cout << " File not found " << endl;
	return; // signal error
      }
      cout << " file opened : " << filename << endl;
      
      char line[500];
      for (int i = 0; i < 3; i++) {
	in_file.getline(line, 500,'\n');
	cout<<line<<endl;
      }
      
      cout << " test map" << endl;
      
      float par0,par1,par2,par3;
      int colid,rowid;
      string name;
      // Read MC tracks
      for(int i=0;i<(52*80);i++)  { // loop over tracks    
	in_file >> par0 >> par1 >> par2 >> par3 >> name >> colid
		>> rowid;
	if (in_file.bad()) { // check for errors
	  cerr << "Cannot read data file" << endl;
	  return;
	}
	if ( in_file.eof() != 0 ) {
	  cerr << in_file.eof() << " " << in_file.gcount() << " "
	       << in_file.fail() << " " << in_file.good() << " end of file "
	       << endl;
	  return;
	}
	
	//cout << " line " << i << " " <<par0<<" "<<par1<<" "<<par2<<" "<<par3<<" "
	//   <<colid<<" "<<rowid<<endl;
	
	CalParameters onePix;
	onePix.p0=par0;
	onePix.p1=par1;
	onePix.p2=par2;
	onePix.p3=par3;
	
	// Convert ROC pixel index to channel 
	int chan = PixelIndices::pixelToChannelROC(rowid,colid);
	calmap.insert(pair<int,CalParameters>(chan,onePix));
	
	// Testing the index conversion, can be skipped 
	pair<int,int> p = PixelIndices::channelToPixelROC(chan);
	if(rowid!=p.first) cout<<" wrong channel row "<<rowid<<" "<<p.first<<endl;
	if(colid!=p.second) cout<<" wrong channel col "<<colid<<" "<<p.second<<endl;
	      
      } // pixel loop in a ROC
 
      cout << " map size  " << calmap.size() <<" max "<<calmap.max_size() << " "
	   <<calmap.empty()<< endl;
      
//     cout << " map size  " << calmap.size()  << endl;
//     map<int,CalParameters,less<int> >::iterator ix,it;
//     map<int,CalParameters,less<int> >::const_iterator ip;
//     for (ix = calmap.begin(); ix != calmap.end(); ++ix) {
//       int i = (*ix).first;
//       pair<int,int> p = channelToPixelROC(i);
//       it  = calmap.find(i);
//       CalParameters y  = (*it).second;
//       CalParameters z = (*ix).second;
//       cout << i <<" "<<p.first<<" "<<p.second<<" "<<y.p0<<" "<<z.p0<<" "<<calmap[i].p0<<endl; 
      
//       //int dummy=0;
//       //cin>>dummy;
//     }

    } // end if readparameters
  } // end if missCalibration 

  LogInfo ("PixelDigitizer ") <<"SiPixelDigitizerAlgorithm constructed"
			      <<"Configuration parameters:" 
			      << "Threshold/Gain = "  
			      << "threshold in electron FPix = " 
			      << theThresholdInE_FPix
			      << "threshold in electron BPix = " 
			      << theThresholdInE_BPix 
			      <<" " << theElectronPerADC << " " << theAdcFullScale 
			      <<" or "<< theAdcFullScaleStack
			      << " The delta cut-off is set to " << tMax
			      << " pix-inefficiency "<<thePixelLuminosity
			      << " Number of Barrel Pixel layers: "<<NumberOfBarrelLayers;

}
//=========================================================================
SiPixelDigitizerAlgorithm::~SiPixelDigitizerAlgorithm() {

  LogDebug ("PixelDigitizer")<<"SiPixelDigitizerAlgorithm deleted";
  
  // Destructor
  delete gaussDistribution_;
  if (gaussDistributionVCALNoise_) delete gaussDistributionVCALNoise_;
  delete flatDistribution_;

  delete theSiPixelGainCalibrationService_;

  // Threshold gaussian smearing:
  if(addThresholdSmearing) {
    delete smearedThreshold_FPix_;
    delete smearedThreshold_BPix_;
    //Carlotta
    delete smearedThreshold_BPix_L1_;
  }

  if(addNoise) delete theNoiser;
  if(fluctuateCharge) delete fluctuate;

}

//=========================================================================
edm::DetSet<PixelDigi>::collection_type 
SiPixelDigitizerAlgorithm::run(
			       const std::vector<PSimHit> &input,
			       PixelGeomDetUnit *pixdet,
			       GlobalVector bfield) {
  
  _detp = pixdet; //cache the PixGeomDetUnit
  _PixelHits=input; //cache the SimHit
  _bfield=bfield; //cache the drift direction

  // Pixel Efficiency moved from the constructor to the method run because
  // the information of the det are not available in the constructor
  // Effciency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi
  
  detID= _detp->geographicalId().rawId();
  
  _signal.clear();
  
  // initalization  of pixeldigisimlinks
  link_coll.clear();
  
  //Digitization of the SimHits of a given pixdet
  vector<PixelDigi> collector =digitize(pixdet);

  // edm::DetSet<PixelDigi> collector;
  
#ifdef TP_DEBUG
  LogDebug ("PixelDigitizer") << "[SiPixelDigitizerAlgorithm] converted " << collector.size() << " PixelDigis in DetUnit" << detID; 
#endif
  
  return collector;
}

//============================================================================
vector<PixelDigi> SiPixelDigitizerAlgorithm::digitize(PixelGeomDetUnit *det){
  
  if( _PixelHits.size() > 0 || addNoisyPixels) {
    
    topol=&det->specificTopology(); // cache topology
    numColumns = topol->ncolumns();  // det module number of cols&rows
    numRows = topol->nrows();
    numROCX = topol->rocsX();
    numROCY = topol->rocsY();
    
    // full detector thickness
    moduleThickness = det->specificSurface().bounds().thickness(); 
    
    // The index converter is only needed when inefficiencies or misscalibration
    // are simulated.
    if((pixelInefficiency>0) || doMissCalibrate ) {  // Init pixel indices
      pIndexConverter = new PixelIndices(numColumns,numRows,numROCX,numROCY);
    }
    
    // Noise already defined in electrons
    //thePixelThresholdInE = thePixelThreshold * theNoiseInElectrons ; 
    // Find the threshold in noise units, needed for the noiser.

  unsigned int Sub_detid=DetId(detID).subdetId();
  //Carlotta
  int lay = 0;
  lay = PXBDetId(detID).layer();

  if(theNoiseInElectrons>0.){
    if(Sub_detid == PixelSubdetector::PixelBarrel){ // Barrel modules
      if(addThresholdSmearing) { 
	//Start of Carlotta's addition
	if (lay == 1)  { thePixelThresholdInE = smearedThreshold_BPix_L1_->fire(); } // gaussian smearing
	else { thePixelThresholdInE = smearedThreshold_BPix_->fire(); } // gaussian smearing
      } else {
	if (lay == 1) { thePixelThresholdInE = theThresholdInE_BPix_L1; } //no smearing
	else { thePixelThresholdInE = theThresholdInE_BPix; } // no smearing
	//End of Carlotta's addition
      }
      
      thePixelThreshold = thePixelThresholdInE/theNoiseInElectrons; 
      
    } else { // Forward disks modules 
      if(addThresholdSmearing) {
	thePixelThresholdInE = smearedThreshold_FPix_->fire(); // gaussian smearing
      } else {
	thePixelThresholdInE = theThresholdInE_FPix; // no smearing
      }
      
      thePixelThreshold = thePixelThresholdInE/theNoiseInElectrons;
      
    }
  } else {
    thePixelThreshold = 0.;
  }


#ifdef TP_DEBUG
    LogDebug ("PixelDigitizer") 
      << " PixelDigitizer "  
      << numColumns << " " << numRows << " " << moduleThickness;
#endif

    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits
 
    vector<PSimHit>::const_iterator ssbegin; 
    for (ssbegin= _PixelHits.begin();ssbegin !=_PixelHits.end(); ++ssbegin) {
      
#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer") 
	<< (*ssbegin).particleType() << " " << (*ssbegin).pabs() << " " 
	<< (*ssbegin).energyLoss() << " " << (*ssbegin).tof() << " " 
	<< (*ssbegin).trackId() << " " << (*ssbegin).processType() << " " 
	<< (*ssbegin).detUnitId()  
	<< (*ssbegin).entryPoint() << " " << (*ssbegin).exitPoint() ; 
#endif      
      
      _collection_points.clear();  // Clear the container
      // fill _collection_points for this SimHit, indpendent of topology
      // Check the TOF cut
      //if (std::abs( (*ssbegin).tof() )<theTofCut){ // old cut
      if ( ((*ssbegin).tof() >= theTofLowerCut) && ((*ssbegin).tof() <= theTofUpperCut) ) {
	primary_ionization(*ssbegin); // fills _ionization_points	
	drift(*ssbegin);  // transforms _ionization_points to _collection_points  	
	// compute induced signal on readout elements and add to _signal
	induce_signal(*ssbegin); // *ihit needed only for SimHit<-->Digi link
      } //  end if 
    } // end for 

    if(use_module_killing_ && !use_deadmodule_DB_) // remove dead modules using the list in cfg file
      module_killing_conf();
  
    if(addNoise) add_noise();  // generate noise
    // Do only if needed 

    if((pixelInefficiency>0) && (_signal.size()>0)) 
      pixel_inefficiency(); // Kill some pixels

    if(use_ineff_from_db_ && (_signal.size()>0))
      pixel_inefficiency_db();

    delete pIndexConverter;

    if(use_module_killing_ && use_deadmodule_DB_) // remove dead modules using DB
      module_killing_DB();

    if(use_module_killing_ && !use_deadmodule_DB_) // remove dead modules using the list in cfg file
      module_killing_conf();

  }

  make_digis();

  return internal_coll;
}

//***********************************************************************/
// Generate primary ionization along the track segment. 
// Divide the track into small sub-segments  
void SiPixelDigitizerAlgorithm::primary_ionization(const PSimHit& hit) {

// Straight line approximation for trajectory inside active media

  const float SegmentLength = 0.0010; //10microns in cm
  float energy;
  
  // Get the 3D segment direction vector 
  LocalVector direction = hit.exitPoint() - hit.entryPoint(); 
  
  float eLoss = hit.energyLoss();  // Eloss in GeV
  float length = direction.mag();  // Track length in Silicon
  
  NumberOfSegments = int ( length / SegmentLength); // Number of segments
  if(NumberOfSegments < 1) NumberOfSegments = 1;
  
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") 
    << " enter primary_ionzation " << NumberOfSegments 
    << " shift = " 
    << (hit.exitPoint().x()-hit.entryPoint().x()) << " " 
    << (hit.exitPoint().y()-hit.entryPoint().y()) << " " 
    << (hit.exitPoint().z()-hit.entryPoint().z()) << " "
    << hit.particleType() <<" "<< hit.pabs() ; 
#endif  

  float* elossVector = new float[NumberOfSegments];  // Eloss vector

  if( fluctuateCharge ) {
    //MP DA RIMUOVERE ASSOLUTAMENTE
    int pid = hit.particleType();
    //int pid=211;  // assume it is a pion
  
    float momentum = hit.pabs();
    // Generate fluctuated charge points
    fluctuateEloss(pid, momentum, eLoss, length, NumberOfSegments, 
		   elossVector);
  }
  
  _ionization_points.resize( NumberOfSegments); // set size

  // loop over segments
  for ( int i = 0; i != NumberOfSegments; i++) {
    // Divide the segment into equal length subsegments 
    Local3DPoint point = hit.entryPoint() + 
      float((i+0.5)/NumberOfSegments) * direction;

    if( fluctuateCharge ) 
      energy = elossVector[i]/GeVperElectron; // Convert charge to elec.
    else
      energy = hit.energyLoss()/GeVperElectron/float(NumberOfSegments);
    
    EnergyDepositUnit edu( energy, point); //define position,energy point
    _ionization_points[i] = edu; // save
    
#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer") 
      << i << " " << _ionization_points[i].x() << " " 
      << _ionization_points[i].y() << " " 
      << _ionization_points[i].z() << " " 
      << _ionization_points[i].energy();
#endif
    
  }  // end for loop 

  delete[] elossVector;

}
//****************************************************************************** 

// Fluctuate the charge comming from a small (10um) track segment.
// Use the G4 routine. For mip pions for the moment.  
void SiPixelDigitizerAlgorithm::fluctuateEloss(int pid, float particleMomentum, 
				      float eloss, float length, 
				      int NumberOfSegs,float elossVector[]) {

  // Get dedx for this track
  float dedx;
  if( length > 0.) dedx = eloss/length;
  else dedx = eloss;

  double particleMass = 139.6; // Mass in MeV, Assume pion
  pid = abs(pid);
  if(pid!=211) {       // Mass in MeV
    if(pid==11)        particleMass = 0.511;        
    else if(pid==13)   particleMass = 105.7;
    else if(pid==321)  particleMass = 493.7;
    else if(pid==2212) particleMass = 938.3;
  }
  // What is the track segment length.
  float segmentLength = length/NumberOfSegs;

  // Generate charge fluctuations.
  float de=0.;
  float sum=0.;
  double segmentEloss = (1000.*eloss)/NumberOfSegs; //eloss in MeV
  for (int i=0;i<NumberOfSegs;i++) {
    //       material,*,   momentum,energy,*, *,  mass
    //myglandz_(14.,segmentLength,2.,2.,dedx,de,0.14);
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV 
    // Returns fluctuated eloss in MeV
    double deltaCutoff = tMax; // the cutoff is sometimes redefined inside, so fix it.
    de = fluctuate->SampleFluctuations(double(particleMomentum*1000.),
				      particleMass, deltaCutoff, 
				      double(segmentLength*10.),
				      segmentEloss )/1000.; //convert to GeV
   
    elossVector[i]=de;
    sum +=de;
  }
  
  if(sum>0.) {  // If fluctuations give eloss>0.
    // Rescale to the same total eloss
    float ratio = eloss/sum;
   
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= ratio*elossVector[ii];
  } else {  // If fluctuations gives 0 eloss
    float averageEloss = eloss/NumberOfSegs;
    for (int ii=0;ii<NumberOfSegs;ii++) elossVector[ii]= averageEloss; 
  }
  return;
}

//*******************************************************************************
// Drift the charge segments to the sensor surface (collection plane)
// Include the effect of E-field and B-field 
void SiPixelDigitizerAlgorithm::drift(const PSimHit& hit){


#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter drift " ;
#endif

  _collection_points.resize( _ionization_points.size()); // set size
  
  LocalVector driftDir=DriftDirection();  // get the charge drift direction
  if(driftDir.z() ==0.) {
    LogWarning("Magnetic field") << " pxlx: drift in z is zero ";
    return;
  }  

  // tangent of Lorentz angle
  //float TanLorenzAngleX = driftDir.x()/driftDir.z(); 
  //float TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();

  float TanLorenzAngleX, TanLorenzAngleY,dir_z, CosLorenzAngleX,
    CosLorenzAngleY;
  if ( alpha2Order) {
    
    TanLorenzAngleX = driftDir.x(); // tangen of Lorentz angle
    TanLorenzAngleY = driftDir.y();
    dir_z = driftDir.z(); // The z drift direction
    CosLorenzAngleX = 1./sqrt(1.+TanLorenzAngleX*TanLorenzAngleX); //cosine
    CosLorenzAngleY = 1./sqrt(1.+TanLorenzAngleY*TanLorenzAngleY); //cosine;
    
  } else{
    
    TanLorenzAngleX = driftDir.x();
    TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();
    dir_z = driftDir.z(); // The z drift direction
    CosLorenzAngleX = 1./sqrt(1.+TanLorenzAngleX*TanLorenzAngleX); //cosine to estimate the path length
    CosLorenzAngleY = 1.;
  }
  
  
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") 
    << " Lorentz Tan " << TanLorenzAngleX << " " << TanLorenzAngleY <<" "
    << CosLorenzAngleX << " " << CosLorenzAngleY << " "
    << moduleThickness*TanLorenzAngleX << " " << driftDir;
#endif  
  
  float Sigma_x = 1.;  // Charge spread 
  float Sigma_y = 1.;
  float DriftDistance; // Distance between charge generation and collection 
  float DriftLength;   // Actual Drift Lentgh
  float Sigma;
  
  
  for (unsigned int i = 0; i != _ionization_points.size(); i++) {
    
    float SegX, SegY, SegZ; // position
    SegX = _ionization_points[i].x();
    SegY = _ionization_points[i].y();
    SegZ = _ionization_points[i].z();
    
    // Distance from the collection plane
    //DriftDistance = (moduleThickness/2. + SegZ); // Drift to -z 
    // Include explixitely the E drift direction (for CMS dir_z=-1)
    DriftDistance = moduleThickness/2. - (dir_z * SegZ); // Drift to -z 
    
    //if( DriftDistance <= 0.) 
    //cout<<" <=0 "<<DriftDistance<<" "<<i<<" "<<SegZ<<" "<<dir_z<<" "
    //  <<SegX<<" "<<SegY<<" "<<(moduleThickness/2)<<" "
    //  <<_ionization_points[i].energy()<<" "
    //  <<hit.particleType()<<" "<<hit.pabs()<<" "<<hit.energyLoss()<<" "
    //  <<hit.entryPoint()<<" "<<hit.exitPoint()
    //  <<endl;

    if( DriftDistance < 0.) {
      DriftDistance = 0.;
    } else if ( DriftDistance > moduleThickness )
      DriftDistance = moduleThickness;
    
    // Assume full depletion now, partial depletion will come later.
    float XDriftDueToMagField = DriftDistance * TanLorenzAngleX;
    float YDriftDueToMagField = DriftDistance * TanLorenzAngleY;
    
    // Shift cloud center
    float CloudCenterX = SegX + XDriftDueToMagField;
    float CloudCenterY = SegY + YDriftDueToMagField;

    // Calculate how long is the charge drift path
    DriftLength = sqrt( DriftDistance*DriftDistance + 
                        XDriftDueToMagField*XDriftDueToMagField +
                        YDriftDueToMagField*YDriftDueToMagField );

    // What is the charge diffusion after this path
    Sigma = sqrt(DriftLength/Dist300) * Sigma0;

    // Project the diffusion sigma on the collection plane
    Sigma_x = Sigma / CosLorenzAngleX ;
    Sigma_y = Sigma / CosLorenzAngleY ;

    SignalPoint sp( CloudCenterX, CloudCenterY,
     Sigma_x, Sigma_y, hit.tof(), _ionization_points[i].energy() );

    // Load the Charge distribution parameters
    _collection_points[i] = (sp);

  } // loop over ionization points, i.
 
} // end drift

//*************************************************************************
// Induce the signal on the collection plane of the active sensor area.
void SiPixelDigitizerAlgorithm::induce_signal( const PSimHit& hit) {

  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)

#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer") 
      << " enter induce_signal, " 
      << topol->pitch().first << " " << topol->pitch().second; //OK
#endif

   // local map to store pixels hit by 1 Hit.      
   typedef map< int, float, less<int> > hit_map_type;
   hit_map_type hit_signal;

   // map to store pixel integrals in the x and in the y directions
   map<int, float, less<int> > x,y; 
   
   // Assign signals to readout channels and store sorted by channel number
   
   // Iterate over collection points on the collection plane
   for ( vector<SignalPoint>::const_iterator i=_collection_points.begin();
	 i != _collection_points.end(); i++) {
     
     float CloudCenterX = i->position().x(); // Charge position in x
     float CloudCenterY = i->position().y(); //                 in y
     float SigmaX = i->sigma_x();            // Charge spread in x
     float SigmaY = i->sigma_y();            //               in y
     float Charge = i->amplitude();          // Charge amplitude


     //if(SigmaX==0 || SigmaY==0) {
     //cout<<SigmaX<<" "<<SigmaY
     //   << " cloud " << i->position().x() << " " << i->position().y() << " " 
     //   << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude()<<endl;
     //}

#ifdef TP_DEBUG
       LogDebug ("Pixel Digitizer") 
	 << " cloud " << i->position().x() << " " << i->position().y() << " " 
	 << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude();
#endif      
     
     // Find the maximum cloud spread in 2D plane , assume 3*sigma
     float CloudRight = CloudCenterX + ClusterWidth*SigmaX;
     float CloudLeft  = CloudCenterX - ClusterWidth*SigmaX;
     float CloudUp    = CloudCenterY + ClusterWidth*SigmaY;
     float CloudDown  = CloudCenterY - ClusterWidth*SigmaY;
     
     // Define 2D cloud limit points
     LocalPoint PointRightUp  = LocalPoint(CloudRight,CloudUp);
     LocalPoint PointLeftDown = LocalPoint(CloudLeft,CloudDown);
     
     // This points can be located outside the sensor area.
     // The conversion to measurement point does not check for that
     // so the returned pixel index might be wrong (outside range).
     // We rely on the limits check below to fix this.
     // But remember whatever we do here THE CHARGE OUTSIDE THE ACTIVE
     // PIXEL ARE IS LOST, it should not be collected.

     // Convert the 2D points to pixel indices
     MeasurementPoint mp = topol->measurementPosition(PointRightUp ); //OK
     
     int IPixRightUpX = int( floor( mp.x()));
     int IPixRightUpY = int( floor( mp.y()));
     
#ifdef TP_DEBUG
     LogDebug ("Pixel Digitizer") << " right-up " << PointRightUp << " " 
				  << mp.x() << " " << mp.y() << " "
				  << IPixRightUpX << " " << IPixRightUpY ;
#endif
 
     mp = topol->measurementPosition(PointLeftDown ); //OK
    
     int IPixLeftDownX = int( floor( mp.x()));
     int IPixLeftDownY = int( floor( mp.y()));
     
#ifdef TP_DEBUG
     LogDebug ("Pixel Digitizer") << " left-down " << PointLeftDown << " " 
				  << mp.x() << " " << mp.y() << " "
				  << IPixLeftDownX << " " << IPixLeftDownY ;
#endif

     // Check detector limits to correct for pixels outside range.
     IPixRightUpX = numRows>IPixRightUpX ? IPixRightUpX : numRows-1 ;
     IPixRightUpY = numColumns>IPixRightUpY ? IPixRightUpY : numColumns-1 ;
     IPixLeftDownX = 0<IPixLeftDownX ? IPixLeftDownX : 0 ;
     IPixLeftDownY = 0<IPixLeftDownY ? IPixLeftDownY : 0 ;
     
     x.clear(); // clear temporary integration array
     y.clear();

     // First integrate cahrge strips in x
     int ix; // TT for compatibility
     for (ix=IPixLeftDownX; ix<=IPixRightUpX; ix++) {  // loop over x index
       float xUB, xLB, UpperBound, LowerBound;
      
       // Why is set to 0 if ix=0, does it meen that we accept charge 
       // outside the sensor? CHeck How it was done in ORCA? 
       //if (ix == 0) LowerBound = 0.;
       if (ix == 0 || SigmaX==0. )  // skip for surface segemnts 
	 LowerBound = 0.;
       else {
	 mp = MeasurementPoint( float(ix), 0.0);
	 xLB = topol->localPosition(mp).x();
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xLB-CloudCenterX)/SigmaX, &result);
	 if (status != 0)  
	   LogWarning ("Integration")<<"could not compute gaussian probability";
	 LowerBound = 1-result.val;
       }
     
       if (ix == numRows-1 || SigmaX==0. ) 
	 UpperBound = 1.;
       else {
	 mp = MeasurementPoint( float(ix+1), 0.0);
	 xUB = topol->localPosition(mp).x();
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xUB-CloudCenterX)/SigmaX, &result);
	 if (status != 0)  
	   LogWarning ("Integration")<<"could not compute gaussian probability";
	 UpperBound = 1. - result.val;
       }
       
       float   TotalIntegrationRange = UpperBound - LowerBound; // get strip
       x[ix] = TotalIntegrationRange; // save strip integral 
       //if(SigmaX==0 || SigmaY==0) 
       //cout<<TotalIntegrationRange<<" "<<ix<<endl;

     }

    // Now integarte strips in y
    int iy; // TT for compatibility
    for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind  
      float yUB, yLB, UpperBound, LowerBound;

      if (iy == 0 || SigmaY==0.) 
	LowerBound = 0.;
      else {
        mp = MeasurementPoint( 0.0, float(iy) );
        yLB = topol->localPosition(mp).y();
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yLB-CloudCenterY)/SigmaY, &result);
	if (status != 0)  
	  LogWarning ("Integration")<<"could not compute gaussian probability";
	LowerBound = 1. - result.val;
      }

      if (iy == numColumns-1 || SigmaY==0. ) 
	UpperBound = 1.;
      else {
        mp = MeasurementPoint( 0.0, float(iy+1) );
        yUB = topol->localPosition(mp).y();
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yUB-CloudCenterY)/SigmaY, &result);
	if (status != 0)  
	  LogWarning ("Integration")<<"could not compute gaussian probability";
	UpperBound = 1. - result.val;
      }

      float   TotalIntegrationRange = UpperBound - LowerBound;
      y[iy] = TotalIntegrationRange; // save strip integral
      //if(SigmaX==0 || SigmaY==0) 
      //cout<<TotalIntegrationRange<<" "<<iy<<endl;
    }       

    // Get the 2D charge integrals by folding x and y strips
    int chan;
    for (ix=IPixLeftDownX; ix<=IPixRightUpX; ix++) {  // loop over x index
      for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind  
	
        float ChargeFraction = Charge*x[ix]*y[iy];

        if( ChargeFraction > 0. ) {
	  chan = PixelDigi::pixelToChannel( ix, iy);  // Get index 
          // Load the amplitude						 
          hit_signal[chan] += ChargeFraction;
	} // endif

	
	mp = MeasurementPoint( float(ix), float(iy) );
	LocalPoint lp = topol->localPosition(mp);
	chan = topol->channel(lp);

#ifdef TP_DEBUG
	LogDebug ("Pixel Digitizer") 
	  << " pixel " << ix << " " << iy << " - "<<" "
	  << chan << " " << ChargeFraction<<" "
	  << mp.x() << " " << mp.y() <<" "
	  << lp.x() << " " << lp.y() << " "  // givex edge position
	  << chan; // edge belongs to previous ?
#endif

      } // endfor iy
    } //endfor ix
   

    // Test conversions (THIS IS FOR TESTING ONLY) comment-out.
    //     mp = topol->measurementPosition( i->position() ); //OK
    //     LocalPoint lp = topol->localPosition(mp);     //OK
    //     pair<float,float> p = topol->pixel( i->position() );  //OK
    //     chan = PixelDigi::pixelToChannel( int(p.first), int(p.second));
    //     pair<int,int> ip = PixelDigi::channelToPixel(chan);
    //     MeasurementPoint mp1 = MeasurementPoint( float(ip.first),
    // 					     float(ip.second) );
    //     LogDebug ("Pixel Digitizer") << " Test "<< mp.x() << " " << mp.y() 
    // 				 << " "<< lp.x() << " " << lp.y() << " "<<" "
    // 				 <<p.first <<" "<<p.second<<" "<<chan<< " "
    // 				 <<" " << ip.first << " " << ip.second << " "
    // 				 << mp1.x() << " " << mp1.y() << " " //OK
    // 				 << topol->localPosition(mp1).x() << " "  //OK
    // 				 << topol->localPosition(mp1).y() << " "
    // 				 << topol->channel( i->position() ); //OK
    
    
  } // loop over charge distributions
   
  // Fill the global map with all hit pixels from this event
   
  for ( hit_map_type::const_iterator im = hit_signal.begin();
	im != hit_signal.end(); im++) {
    _signal[(*im).first] += Amplitude( (*im).second, &hit, (*im).second);
    
    int chan =  (*im).first; 
    pair<int,int> ip = PixelDigi::channelToPixel(chan);

#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer") 
      << " pixel " << ip.first << " " << ip.second << " "
      << _signal[(*im).first];    
#endif
  }

} // end induce_signal

/***********************************************************************/

// Build pixels, check threshold, add misscalibration, ... 
void SiPixelDigitizerAlgorithm::make_digis() {
  internal_coll.reserve(50); internal_coll.clear();

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " make digis "<<" "
			       << " pixel threshold FPix" << theThresholdInE_FPix << " " 
                               << " pixel threshold BPix" << theThresholdInE_BPix << " "
			       << " List pixels passing threshold ";
#endif  
  
  // Loop over hit pixels

  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
    
        float signalInElectrons = (*i).second ;   // signal in electrons

    // Do the miss calibration for calibration studies only.
    //if(doMissCalibrate) signalInElectrons = missCalibrate(signalInElectrons)

    // Do only for pixels above threshold

       if ( signalInElectrons >= thePixelThresholdInE) { // check threshold
             
      int chan =  (*i).first;  // channel number
      pair<int,int> ip = PixelDigi::channelToPixel(chan);
      int adc=0;  // ADC count as integer

      // Do the miss calibration for calibration studies only.
      if(doMissCalibrate) {
	int row = ip.first;  // X in row
	int col = ip.second; // Y is in col
	adc = int(missCalibrate(col,row,signalInElectrons)); //full misscalib.
      } else { // Just do a simple electron->adc conversion
	adc = int( signalInElectrons / theElectronPerADC ); // calibrate gain
      }
      unsigned int Subid=DetId(detID).subdetId();
      int layerIndex=0;
      if (Subid==  PixelSubdetector::PixelBarrel){layerIndex=PXBDetId(detID).layer();}
      adc = min(adc, theAdcFullScale); // Check maximum value

      if (layerIndex>=theFirstStackLayer) {
        // Set to 1 if over the threshold
        if (theAdcFullScaleStack==1) {adc=1;}
        // Make it a linear fit to the full scale of the normal adc count.   Start new adc from 1 not zero.
        if (theAdcFullScaleStack!=1&&theAdcFullScaleStack!=theAdcFullScale) {adc = int (1 + adc * (theAdcFullScaleStack-1)/float(theAdcFullScale) );}
      }

#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer") 
	<< (*i).first << " " << (*i).second << " " << signalInElectrons 
	<< " " << adc << ip.first << " " << ip.second ;
#endif
           
      // Load digis
      internal_coll.push_back( PixelDigi( ip.first, ip.second, adc));
  
      //digilink     
      if((*i).second.hits().size()>0){
	simi.clear();
	unsigned int il=0;
	for( vector<const PSimHit*>::const_iterator ihit = (*i).second.hits().begin();
	     ihit != (*i).second.hits().end(); ihit++) {
	  simi[(**ihit).trackId()].push_back((*i).second.individualampl()[il]);
	  il++;
	}
	
	//sum the contribution of the same trackid 
	for( simlink_map::iterator simiiter=simi.begin();
	     simiiter!=simi.end();
	     simiiter++){
	  
	  float sum_samechannel=0;
	  for (unsigned int iii=0;iii<(*simiiter).second.size();iii++){
	    sum_samechannel+=(*simiiter).second[iii];
	  }
	  float fraction=sum_samechannel/(*i).second;
	  if (fraction>1.) fraction=1.;
	  link_coll.push_back(PixelDigiSimLink((*i).first,(*simiiter).first,((*i).second.hits().front())->eventId(),fraction));
	}
	
      }
    }
    
  }
}

/***********************************************************************/

//  Add electronic noise to pixel charge
void SiPixelDigitizerAlgorithm::add_noise() {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter add_noise " << theNoiseInElectrons;
#endif
 
  // First add noise to hit pixels
  
  for ( signal_map_iterator i = _signal.begin(); i != _signal.end(); i++) {
    
         if(addChargeVCALSmearing) 
      {
	float theSmearedChargeRMS=0.;

	if((*i).second < 3000)
	  {
	    theSmearedChargeRMS = 543.6 - (*i).second * 0.093;
	  } else if((*i).second < 6000){
	    theSmearedChargeRMS = 307.6 - (*i).second * 0.01;
	  } else{
	    theSmearedChargeRMS = -432.4 +(*i).second * 0.123;
	}

	// Noise from Vcal smearing:
	float noise_ChargeVCALSmearing = theSmearedChargeRMS * gaussDistributionVCALNoise_->fire() ;
	(*i).second += Amplitude( noise_ChargeVCALSmearing,0,-1.);

	// Noise from full readout:
	float noise  = gaussDistribution_->fire() ;

		if(((*i).second + Amplitude(noise+noise_ChargeVCALSmearing,0,-1.)) < 0. ) {
		  (*i).second.set(0);}
		else{
	(*i).second +=Amplitude(noise+noise_ChargeVCALSmearing,0,-1.);
		}

      } // End if addChargeVCalSmearing
	 else
     {
	// Noise: ONLY full READOUT Noise.
	// Use here the FULL readout noise, including TBM,ALT,AOH,OPT-REC.

	float noise  = gaussDistribution_->fire() ;
		if(((*i).second + Amplitude(noise,0,-1.)) < 0. ) {
		  (*i).second.set(0);}
		else{
	(*i).second +=Amplitude(noise,0,-1.);
		}
     } // end if only Noise from full readout
    
  }

  if(!addNoisyPixels)  // Option to skip noise in non-hit pixels
    return;

  // Add noise on non-hit pixels
  // Use here the pixel noise 
  int numberOfPixels = (numRows * numColumns);
  map<int,float, less<int> > otherPixels;
  map<int,float, less<int> >::iterator mapI;

  theNoiser->generate(numberOfPixels, 
                      thePixelThreshold, //thr. in un. of nois
		      theNoiseInElectrons, // noise in elec. 
                      otherPixels );

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") 
    <<  " Add noisy pixels " << numRows << " " 
    << numColumns << " " << theNoiseInElectrons << " " 
    << theThresholdInE_FPix << theThresholdInE_BPix <<" "<< numberOfPixels<<" " 
    << otherPixels.size() ;
#endif
  
  // Add noisy pixels
  for (mapI = otherPixels.begin(); mapI!= otherPixels.end(); mapI++) {
    int iy = ((*mapI).first) / numRows;
    int ix = ((*mapI).first) - (iy*numRows);

    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) ) 
      LogWarning ("Pixel Geometry") << " error in iy " << iy ;
    if( ix < 0 || ix > (numRows-1) )
      LogWarning ("Pixel Geometry")  << " error in ix " << ix ;

    int chan = PixelDigi::pixelToChannel(ix, iy);

#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer")
      <<" Storing noise = " << (*mapI).first << " " << (*mapI).second 
      << " " << ix << " " << iy << " " << chan ;
#endif
 
    if(_signal[chan] == 0){
      //      float noise = float( (*mapI).second );
      int noise=int( (*mapI).second );
      _signal[chan] = Amplitude (noise, 0,-1.);
    }
  }

}

/***********************************************************************/

// Simulate the readout inefficiencies. 
// Delete a selected number of single pixels, dcols and rocs.
void SiPixelDigitizerAlgorithm::pixel_inefficiency() {


  // Predefined efficiencies
  float pixelEfficiency  = 1.0;
  float columnEfficiency = 1.0;
  float chipEfficiency   = 1.0;

  // setup the chip indices conversion
  unsigned int Subid=DetId(detID).subdetId();
  if    (Subid==  PixelSubdetector::PixelBarrel){// barrel layers
    int layerIndex=PXBDetId(detID).layer();
    pixelEfficiency  = thePixelEfficiency[layerIndex-1];
    columnEfficiency = thePixelColEfficiency[layerIndex-1];
    chipEfficiency   = thePixelChipEfficiency[layerIndex-1];
 
    // already initialized in calling routine (digitize) and deleted there after this call
    //pIndexConverter = new PixelIndices(numColumns,numRows, numROCX, numROCY);  //--Hec: [Sep-09] Initialize variables
    int ax = pIndexConverter->GetdefaultDetSizeInX();  //--Hec:
    int ay = pIndexConverter->GetdefaultDetSizeInY();  //--Hec:

    // This should never happen
    if(numColumns>ay)  LogWarning ("Pixel Geometry") <<" wrong columns in barrel "<<numColumns<<" > "<<ay;     //--Hec:
    if(numRows>ax)  LogWarning ("Pixel Geometry") <<" wrong rows in barrel "<<numRows<<" > "<<ax;     //--Hec:
    
  } else {                // forward disks
   
    // For endcaps take same for each endcap
    pixelEfficiency  = thePixelEfficiency[NumberOfBarrelLayers];
    columnEfficiency = thePixelColEfficiency[NumberOfBarrelLayers];
    chipEfficiency   = thePixelChipEfficiency[NumberOfBarrelLayers];
 
    int ax = pIndexConverter->GetdefaultDetSizeInX();  // numRow max is 160, but use num from geometry
    int ay = pIndexConverter->GetdefaultDetSizeInY();  // numCol max is 416, but use num from geometry
    // Sometimes the forward pixels have wrong size, 
    // this crashes the index conversion, so exit.
    if(numColumns>ay || numRows>ax) {
      if(numColumns>ay)  LogWarning ("Pixel Geometry") <<" wrong columns in endcaps "<<numColumns<<" > "<<ay;
      if(numRows>ax)  LogWarning ("Pixel Geometry") <<" wrong rows in endcaps "<<numRows<<" > "<<ax;
      return;
    }
  } // if barrel/forward

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter pixel_inefficiency " << pixelEfficiency << " " 
			       << columnEfficiency << " " << chipEfficiency;
#endif
  
  // Initilize the index converter
  //PixelIndices indexConverter(numColumns,numRows);
  int chipIndex = 0;
  int rowROC = 0;
  int colROC = 0;
  map<int, int, less<int> >chips, columns;
  map<int, int, less<int> >::iterator iter;
  
  // Find out the number of columns and rocs hits
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for (signal_map_iterator i = _signal.begin();i != _signal.end();i++) {
    
    int chan = i->first;
    pair<int,int> ip = PixelDigi::channelToPixel(chan);
    int row = ip.first;  // X in row
    int col = ip.second; // Y is in col
    //transform to ROC index coordinates   
    pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);
    int dColInChip = pIndexConverter->DColumn(colROC); // get ROC dcol from ROC col 
    //dcol in mod
    int dColInDet = pIndexConverter->DColumnInModule(dColInChip,chipIndex); 
      
    chips[chipIndex]++;
    columns[dColInDet]++;
  }
  
  // Delete some ROC hits.
  for ( iter = chips.begin(); iter != chips.end() ; iter++ ) {
    //float rand  = RandFlat::shoot();
    float rand  = flatDistribution_->fire();
    if( rand > chipEfficiency ) chips[iter->first]=0;
  }

  // Delete some Dcol hits.
  for ( iter = columns.begin(); iter != columns.end() ; iter++ ) {
    //float rand  = RandFlat::shoot();
    float rand  = flatDistribution_->fire();
    if( rand > columnEfficiency ) columns[iter->first]=0;
  }
  
  // Now loop again over pixels to kill some of them.
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {    

    //    int chan = i->first;
    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    int row = ip.first;  // X in row
    int col = ip.second; // Y is in col
    //transform to ROC index coordinates   
    pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);
    int dColInChip = pIndexConverter->DColumn(colROC); //get ROC dcol from ROC col 
    //dcol in mod
    int dColInDet = pIndexConverter->DColumnInModule(dColInChip,chipIndex); 


    //float rand  = RandFlat::shoot();
    float rand  = flatDistribution_->fire();
    if( chips[chipIndex]==0 || columns[dColInDet]==0 
	|| rand>pixelEfficiency ) {
      // make pixel amplitude =0, pixel will be lost at clusterization    
      i->second.set(0.); // reset amplitude, 
    } // end if

  } // end pixel loop
  
} // end pixel_indefficiency

//***********************************************************************

// Fluctuate the gain and offset for the amplitude calibration
// Use gaussian smearing.
//float SiPixelDigitizerAlgorithm::missCalibrate(const float amp) const {
  //float gain  = RandGaussQ::shoot(1.,theGainSmearing);
  //float offset  = RandGaussQ::shoot(0.,theOffsetSmearing);
  //float newAmp = amp * gain + offset;
  // More complex misscalibration 
float SiPixelDigitizerAlgorithm::missCalibrate(int col,int row,
				 const float signalInElectrons) const {

  // Central values
  //const float p0=0.00352, p1=0.868, p2=112., p3=113.; // pix(0,0,0)
  //  const float p0=0.00382, p1=0.886, p2=112.7, p3=113.0; // average roc=0
  //const float p0=0.00492, p1=1.998, p2=90.6, p3=134.1; // average roc=6
  // Smeared (rms)
  //const float s0=0.00020, s1=0.051, s2=5.4, s3=4.4; // average roc=0
  //const float s0=0.00015, s1=0.043, s2=3.2, s3=3.1; // col average roc=0

  // Make 2 sets of parameters for Fpix and BPIx:

  float p0=0.0;
  float p1=0.0;
  float p2=0.0;
  float p3=0.0;

  unsigned int Sub_detid=DetId(detID).subdetId();

    if (Sub_detid == PixelSubdetector::PixelBarrel){// barrel layers
      p0 = BPix_p0;
      p1 = BPix_p1;
      p2 = BPix_p2;
      p3 = BPix_p3;
    } else {// forward disks
      p0 = FPix_p0;
      p1 = FPix_p1;
      p2 = FPix_p2;
      p3 = FPix_p3;
    }

  //  const float electronsPerVCAL = 65.5; // our present VCAL calibration (feb 2009)
  //  const float electronsPerVCAL_Offset = -414.0; // our present VCAL calibration (feb 2009)
  float newAmp = 0.; //Modified signal

  // Convert electrons to VCAL units
  float signal = (signalInElectrons-electronsPerVCAL_Offset)/electronsPerVCAL;

  // Simulate the analog response with fixed parametrization
  newAmp = p3 + p2 * tanh(p0*signal - p1);


  // Use the pixel-by-pixel calibrations
  //transform to ROC index coordinates
  //int chipIndex=0, colROC=0, rowROC=0;
  //pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);

  // Use calibration from a file
  //int chanROC = PixelIndices::pixelToChannelROC(rowROC,colROC); // use ROC coordinates
  //float pp0=0, pp1=0,pp2=0,pp3=0;
  //map<int,CalParameters,less<int> >::const_iterator it=calmap.find(chanROC);
  //CalParameters y  = (*it).second;
  //pp0 = y.p0;
  //pp1 = y.p1;
  //pp2 = y.p2;
  //pp3 = y.p3;

  //
  // Use random smearing 
  // Randomize the pixel response
  //float pp0  = RandGaussQ::shoot(p0,s0);
  //float pp1  = RandGaussQ::shoot(p1,s1);
  //float pp2  = RandGaussQ::shoot(p2,s2);
  //float pp3  = RandGaussQ::shoot(p3,s3);

  //newAmp = pp3 + pp2 * tanh(pp0*signal - pp1); // Final signal

  //cout<<" misscalibrate "<<col<<" "<<row<<" "<<chipIndex<<" "<<colROC<<" "
  //  <<rowROC<<" "<<signalInElectrons<<" "<<signal<<" "<<newAmp<<" "
  //  <<(signalInElectrons/theElectronPerADC)<<endl;

  return newAmp;
}  
//******************************************************************************

// Set the drift direction accoring to the Bfield in local det-unit frame
// Works for both barrel and forward pixels.
// Replace the sign convention to fit M.Swartz's formulaes.
// Configurations for barrel and foward pixels possess different tanLorentzAngleperTesla 
// parameter value

LocalVector SiPixelDigitizerAlgorithm::DriftDirection(){
  Frame detFrame(_detp->surface().position(),_detp->surface().rotation());
  LocalVector Bfield=detFrame.toLocal(_bfield);
  
  float alpha2_FPix;
  float alpha2_BPix;
  float alpha2;

  //float dir_x = -tanLorentzAnglePerTesla * Bfield.y();
  //float dir_y = +tanLorentzAnglePerTesla * Bfield.x();
  //float dir_z = -1.; // E field always in z direction, so electrons go to -z
  // The dir_z has to be +/- 1. !
  // LocalVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);

  float dir_x = 0.0;
  float dir_y = 0.0;
  float dir_z = 0.0;
  float scale = 0.0;
  
  unsigned int Sub_detid=DetId(detID).subdetId();

  // Read Lorentz angle from cfg file:**************************************************************
  
  if(!use_LorentzAngle_DB_){ 
    
    if ( alpha2Order) {
      alpha2_FPix = tanLorentzAnglePerTesla_FPix*tanLorentzAnglePerTesla_FPix;
      alpha2_BPix = tanLorentzAnglePerTesla_BPix*tanLorentzAnglePerTesla_BPix;
    }else {
      alpha2_FPix = 0.0;
      alpha2_BPix = 0.0;
    }
    
    if (Sub_detid == PixelSubdetector::PixelBarrel){// barrel layers
      dir_x = -( tanLorentzAnglePerTesla_BPix * Bfield.y() + alpha2_BPix* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_BPix * Bfield.x() - alpha2_BPix* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_BPix* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_BPix* Bfield.z()*Bfield.z() );
      
    } else {// forward disks
      dir_x = -( tanLorentzAnglePerTesla_FPix * Bfield.y() + alpha2_FPix* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_FPix * Bfield.x() - alpha2_FPix* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_FPix* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_FPix* Bfield.z()*Bfield.z() );
    }
    } // end: Read LA from cfg file.
  
  //Read Lorentz angle from DB:********************************************************************
  if(use_LorentzAngle_DB_){ 
    std::map<unsigned int,float> detid_la= SiPixelLorentzAngle_->getLorentzAngles();
    std::map<unsigned int,float>::const_iterator it;
    
    
    for (it=detid_la.begin();it!=detid_la.end();it++)
    {
      if (detID==it->first) {
	if (alpha2Order) {
	  alpha2 = it->second * it->second;
	}  
	else {
	  alpha2 = 0.0;
	} 
	//	std::cout << "detID is: " << it->first << "The LA per tesla is: " << it->second << std::endl;
	dir_x = -( it->second * Bfield.y() + alpha2 * Bfield.z()* Bfield.x() );
	dir_y = +( it->second * Bfield.x() - alpha2 * Bfield.z()* Bfield.y() );
	dir_z = -(1 + alpha2 * Bfield.z()*Bfield.z() );
	scale = (1 + alpha2 * Bfield.z()*Bfield.z() );
      }
    } 
  }// end: Read LA from DataBase.
  
  LocalVector theDriftDirection = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );  
  
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " The drift direction in local coordinate is "   
			       << theDriftDirection ;
#endif
  
  return theDriftDirection;
}

//****************************************************************************************************

void SiPixelDigitizerAlgorithm::pixel_inefficiency_db(void){
  if(!use_ineff_from_db_)
    return;
  
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {    
    
    //    int chan = i->first;
    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    int row = ip.first;  // X in row
    int col = ip.second; // Y is in col
    uint32_t detid = detID;
    //transform to ROC index coordinates   
    if(theSiPixelGainCalibrationService_->isDead(detid, col, row)){
      //      std::cout << "now in isdead check, row " << detid << " " << col << "," << row << std::endl;
      // make pixel amplitude =0, pixel will be lost at clusterization    
      i->second.set(0.); // reset amplitude, 
    } // end if
  } // end pixel loop
} // end pixel_indefficiency


//****************************************************************************************************

void SiPixelDigitizerAlgorithm::module_killing_conf(void){
  if(!use_module_killing_)
    return;
  
  bool isbad=false;
  int detid = detID;
  
  Parameters::iterator itDeadModules=DeadModules.begin();
  
  for(; itDeadModules != DeadModules.end(); ++itDeadModules){
    int Dead_detID = itDeadModules->getParameter<int>("Dead_detID"); 
    if(detid==Dead_detID){
      isbad=true;
      break;
    }
  }
  
  if(!isbad)
    return;
  
  std::string Module = itDeadModules->getParameter<std::string>("Module");
  
  if(Module=="whole"){
    for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {    
      i->second.set(0.); // reset amplitude
    }
  }
  
  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {    
    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    
    if(Module=="tbmA" && ip.first>=80 && ip.first<=159){
      i->second.set(0.);
    }
    
    if( Module=="tbmB" && ip.first<=79){
      i->second.set(0.);

    }

  }
}


//****************************************************************************************************
void SiPixelDigitizerAlgorithm::module_killing_DB(void){
  if(!use_module_killing_)
    return;
  
  bool isbad=false;
  uint32_t detid = detID;
  
  std::vector<SiPixelQuality::disabledModuleType>disabledModules = SiPixelBadModule_->getBadComponentList();

  SiPixelQuality::disabledModuleType badmodule;

  for (size_t id=0;id<disabledModules.size();id++)
    {
      if(detid==disabledModules[id].DetID){
	isbad=true;
        badmodule = disabledModules[id];
	break;
      }
    }
  
  if(!isbad)
    return;

  if(badmodule.errorType == 0){ // this is a whole dead module.

    for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {
           i->second.set(0.); // reset amplitude
    }
  }
  else { // all other module types: half-modules and single ROCs.
	   // Get Bad ROC position:
	   //follow the example of getBadRocPositions in CondFormats/SiPixelObjects/src/SiPixelQuality.cc 
  std::vector<GlobalPixel> badrocpositions (0);
  std::pair<uint8_t, uint8_t> coord(1,1);
   for(unsigned int j = 0; j < 16; j++){
     if (SiPixelBadModule_->IsRocBad(detid, j) == true){

    std::vector<CablingPathToDetUnit> path = map_.product()->pathToDetUnit(detid);
    typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
      for  (IT it = path.begin(); it != path.end(); ++it) {
          const PixelROC* myroc = map_.product()->findItem(*it);
          if( myroc->idInDetUnit() == j) {
	                  LocalPixel::RocRowCol  local = { 39, 25};   //corresponding to center of ROC row, col
              GlobalPixel global = myroc->toGlobal( LocalPixel(local) );
	      badrocpositions.push_back(global);
         break;
       }
      }
     }
   }// end of getBadRocPositions

  for(signal_map_iterator i = _signal.begin();i != _signal.end(); i++) {    
    pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos

  for(std::vector<GlobalPixel>::const_iterator it = badrocpositions.begin(); it != badrocpositions.end(); ++it){
    //  std::cout<<"getBadRocPositions, detidnumber: " << detid << "row is "<<it->row<<" col is "<<it->col<<std::endl;
  if(it->row==120){
  if((ip.first>=(it->row)-40  && ip.first<=(it->row)+39) && (ip.second>=(it->col)-25 && ip.second<=(it->col)+26)){
		     i->second.set(0.);
  }
  }
  else {
  if((ip.first>=(it->row)-39  && ip.first<=(it->row)+40) && (ip.second>=(it->col)-25 && ip.second<=(it->col)+26)){
		     i->second.set(0.);
       }
          }
  }
  }
  }
}

