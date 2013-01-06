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
// February, 2011: Time improvement in DriftDirection()  (J. Bashir Butt)
// June, 2011: Bug Fix for pixels on ROC edges in module_killing_DB() (J. Bashir Butt)
#include <iostream>

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SiPixelDigitizerAlgorithm.h"

#include <gsl/gsl_sf_erf.h>
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

//#include "PixelIndices.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

// Accessing dead pixel modules from the DB:
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"

#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
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

using namespace edm;
using namespace sipixelobjects;

#define TP_DEBUG // protect all LogDebug with ifdef. Takes too much CPU


void SiPixelDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if(use_ineff_from_db_){// load gain calibration service fromdb...
    theSiPixelGainCalibrationService_->setESObjects( es );
  }
  if(use_deadmodule_DB_) {
    es.get<SiPixelQualityRcd>().get(SiPixelBadModule_);
  }
  if(use_LorentzAngle_DB_) {
    // Get Lorentz angle from DB record
    es.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);
  }
  //gets the map and geometry from the DB (to kill ROCs)
  es.get<SiPixelFedCablingMapRcd>().get(map_);
  es.get<TrackerDigiGeometryRecord>().get(geom_);
}

//=========================================================================

SiPixelDigitizerAlgorithm::SiPixelDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng) :

  _signal(),
  makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", true)),
  use_ineff_from_db_(conf.getParameter<bool>("useDB")),
  use_module_killing_(conf.getParameter<bool>("killModules")), // boolean to kill or not modules
  use_deadmodule_DB_(conf.getParameter<bool>("DeadModules_DB")), // boolean to access dead modules from DB
  use_LorentzAngle_DB_(conf.getParameter<bool>("LorentzAngle_DB")), // boolean to access Lorentz angle from DB

  DeadModules(use_deadmodule_DB_ ? Parameters() : conf.getParameter<Parameters>("DeadModules")), // get dead module from cfg file

  // Common pixel parameters
  // These are parameters which are not likely to be changed
  GeVperElectron(3.61E-09), // 1 electron(3.61eV, 1keV(277e, mod 9/06 d.k.
  Sigma0(0.00037),           // Charge diffusion constant 7->3.7
  Dist300(0.0300),          //   normalized to 300micron Silicon
  alpha2Order(conf.getParameter<bool>("Alpha2Order")),   // switch on/off of E.B effect
  ClusterWidth(3.),     // Charge integration spread on the collection plane

  // get external parameters:
  // To account for upgrade geometries do not assume the number 
  // of layers or disks.
  NumberOfBarrelLayers(conf.exists("NumPixelBarrel")?conf.getParameter<int>("NumPixelBarrel"):3),
  NumberOfEndcapDisks(conf.exists("NumPixelEndcap")?conf.getParameter<int>("NumPixelEndcap"):2),

  // ADC calibration 1adc count(135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc](2[adc/kev]
  // Be carefull, this parameter is also used in SiPixelDet.cc to
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC(conf.getParameter<double>("ElectronPerAdc")),

  // ADC saturation value, 255(8bit adc.
  //theAdcFullScale(conf.getUntrackedParameter<int>("AdcFullScale",255)),
  theAdcFullScale(conf.getParameter<int>("AdcFullScale")),
  theAdcFullScaleStack(conf.exists("AdcFullScaleStack")?conf.getParameter<int>("AdcFullScaleStack"):255),
  theFirstStackLayer(conf.exists("FirstStackLayer")?conf.getParameter<int>("FirstStackLayer"):5),

  // Noise in electrons:
  // Pixel cell noise, relevant for generating noisy pixels
  theNoiseInElectrons(conf.getParameter<double>("NoiseInElectrons")),

  // Fill readout noise, including all readout chain, relevant for smearing
  //theReadoutNoise(conf.getUntrackedParameter<double>("ReadoutNoiseInElec",500.)),
  theReadoutNoise(conf.getParameter<double>("ReadoutNoiseInElec")),

  // Pixel threshold in units of noise:
  // thePixelThreshold(conf.getParameter<double>("ThresholdInNoiseUnits")),
  // Pixel threshold in electron units.
  theThresholdInE_FPix(conf.getParameter<double>("ThresholdInElectrons_FPix")),
  theThresholdInE_BPix(conf.getParameter<double>("ThresholdInElectrons_BPix")),
  theThresholdInE_BPix_L1(conf.exists("ThresholdInElectrons_BPix_L1")?conf.getParameter<double>("ThresholdInElectrons_BPix_L1"):theThresholdInE_BPix),

  // Add threshold gaussian smearing:
  theThresholdSmearing_FPix(conf.getParameter<double>("ThresholdSmearing_FPix")),
  theThresholdSmearing_BPix(conf.getParameter<double>("ThresholdSmearing_BPix")),
  theThresholdSmearing_BPix_L1(conf.exists("ThresholdSmearing_BPix_L1")?conf.getParameter<double>("ThresholdSmearing_BPix_L1"):theThresholdSmearing_BPix),

  // electrons to VCAL conversion needed in misscalibrate()
  electronsPerVCAL(conf.getParameter<double>("ElectronsPerVcal")),
  electronsPerVCAL_Offset(conf.getParameter<double>("ElectronsPerVcal_Offset")),

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  //theTofCut(conf.getUntrackedParameter<double>("TofCut",12.5)),
  theTofLowerCut(conf.getParameter<double>("TofLowerCut")),
  theTofUpperCut(conf.getParameter<double>("TofUpperCut")),

  // Get the Lorentz angle from the cfg file:
  tanLorentzAnglePerTesla_FPix(use_LorentzAngle_DB_ ? 0.0 : conf.getParameter<double>("TanLorentzAnglePerTesla_FPix")),
  tanLorentzAnglePerTesla_BPix(use_LorentzAngle_DB_ ? 0.0 : conf.getParameter<double>("TanLorentzAnglePerTesla_BPix")),

  // signal response new parameterization: split Fpix and BPix
  FPix_p0(conf.getParameter<double>("FPix_SignalResponse_p0")),
  FPix_p1(conf.getParameter<double>("FPix_SignalResponse_p1")),
  FPix_p2(conf.getParameter<double>("FPix_SignalResponse_p2")),
  FPix_p3(conf.getParameter<double>("FPix_SignalResponse_p3")),

  BPix_p0(conf.getParameter<double>("BPix_SignalResponse_p0")),
  BPix_p1(conf.getParameter<double>("BPix_SignalResponse_p1")),
  BPix_p2(conf.getParameter<double>("BPix_SignalResponse_p2")),
  BPix_p3(conf.getParameter<double>("BPix_SignalResponse_p3")),

  // Add noise
  addNoise(conf.getParameter<bool>("AddNoise")),

  // Smear the pixel charge with a gaussian which RMS is a function of the
  // pixel charge (Danek's study)
  addChargeVCALSmearing(conf.getParameter<bool>("ChargeVCALSmearing")),

  // Add noisy pixels
  addNoisyPixels(conf.getParameter<bool>("AddNoisyPixels")),

  // Fluctuate charge in track subsegments
  fluctuateCharge(conf.getUntrackedParameter<bool>("FluctuateCharge",true)),

  // Control the pixel inefficiency
  AddPixelInefficiency(conf.getParameter<bool>("AddPixelInefficiencyFromPython")),

  // Add threshold gaussian smearing:
  addThresholdSmearing(conf.getParameter<bool>("AddThresholdSmearing")),

  // Get the constants for the miss-calibration studies
  doMissCalibrate(conf.getParameter<bool>("MissCalibrate")), // Enable miss-calibration
  theGainSmearing(conf.getParameter<double>("GainSmearing")), // sigma of the gain smearing
  theOffsetSmearing(conf.getParameter<double>("OffsetSmearing")), //sigma of the offset smearing

  // delta cutoff in MeV, has to be same as in OSCAR(0.030/cmsim=1.0 MeV
  //tMax(0.030), // In MeV.
  //tMax(conf.getUntrackedParameter<double>("DeltaProductionCut",0.030)),
  tMax(conf.getParameter<double>("DeltaProductionCut")),

  fluctuate(fluctuateCharge ? new SiG4UniversalFluctuation(eng) : 0),
  theNoiser(addNoise ? new GaussianTailNoiseGenerator(eng) : 0),
  calmap(doMissCalibrate ? initCal() : std::map<int,CalParameters,std::less<int> >()),
  theSiPixelGainCalibrationService_(use_ineff_from_db_ ? new SiPixelGainCalibrationOfflineSimService(conf) : 0),
  pixelEfficiencies_(conf, AddPixelInefficiency,NumberOfBarrelLayers,NumberOfEndcapDisks),
  flatDistribution_((addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) ? new CLHEP::RandFlat(eng, 0., 1.) : 0),
  gaussDistribution_((addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) ? new CLHEP::RandGaussQ(eng, 0., theReadoutNoise) : 0),
  gaussDistributionVCALNoise_((addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) ? new CLHEP::RandGaussQ(eng, 0., 1.) : 0),
  // Threshold smearing with gaussian distribution:
  smearedThreshold_FPix_(addThresholdSmearing ? new CLHEP::RandGaussQ(eng, theThresholdInE_FPix , theThresholdSmearing_FPix) : 0),
  smearedThreshold_BPix_(addThresholdSmearing ? new CLHEP::RandGaussQ(eng, theThresholdInE_BPix , theThresholdSmearing_BPix) : 0),
  smearedThreshold_BPix_L1_(addThresholdSmearing ? new CLHEP::RandGaussQ(eng, theThresholdInE_BPix_L1 , theThresholdSmearing_BPix_L1) : 0)
{
  LogInfo ("PixelDigitizer ") <<"SiPixelDigitizerAlgorithm constructed"
			      <<"Configuration parameters:"
			      << "Threshold/Gain = "
			      << "threshold in electron FPix = "
			      << theThresholdInE_FPix
			      << "threshold in electron BPix = "
			      << theThresholdInE_BPix
                              << "threshold in electron BPix Layer1 = "
                              << theThresholdInE_BPix_L1
			      <<" " << theElectronPerADC << " " << theAdcFullScale
			      << " The delta cut-off is set to " << tMax
			      << " pix-inefficiency "<<AddPixelInefficiency;

}

std::map<int, SiPixelDigitizerAlgorithm::CalParameters, std::less<int> >
SiPixelDigitizerAlgorithm::initCal() const {

  using std::cerr;
  using std::cout;
  using std::endl;

  std::map<int, SiPixelDigitizerAlgorithm::CalParameters, std::less<int> > calmap;
  // Prepare for the analog amplitude miss-calibration
  LogDebug ("PixelDigitizer ")
    << " miss-calibrate the pixel amplitude ";

  const bool ReadCalParameters = false;
  if(ReadCalParameters) {   // Read the calibration files from file
    // read the calibration constants from a file (testing only)
    std::ifstream in_file;  // data file pointer
    char filename[80] = "phCalibrationFit_C0.dat";

    in_file.open(filename, std::ios::in ); // in C++
    if(in_file.bad()) {
      cout << " File not found " << endl;
      return calmap; // signal error
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
    std::string name;
    // Read MC tracks
    for(int i=0;i<(52*80);i++)  { // loop over tracks
      in_file >> par0 >> par1 >> par2 >> par3 >> name >> colid >> rowid;
      if(in_file.bad()) { // check for errors
        cerr << "Cannot read data file" << endl;
        return calmap;
      }
      if( in_file.eof() != 0 ) {
        cerr << in_file.eof() << " " << in_file.gcount() << " "
             << in_file.fail() << " " << in_file.good() << " end of file "
             << endl;
        return calmap;
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
      calmap.insert(std::pair<int,CalParameters>(chan,onePix));

      // Testing the index conversion, can be skipped
      std::pair<int,int> p = PixelIndices::channelToPixelROC(chan);
      if(rowid!=p.first) cout<<" wrong channel row "<<rowid<<" "<<p.first<<endl;
      if(colid!=p.second) cout<<" wrong channel col "<<colid<<" "<<p.second<<endl;

    } // pixel loop in a ROC

    cout << " map size  " << calmap.size() <<" max "<<calmap.max_size() << " "
         <<calmap.empty()<< endl;

//     cout << " map size  " << calmap.size()  << endl;
//     map<int,CalParameters,std::less<int> >::iterator ix,it;
//     map<int,CalParameters,std::less<int> >::const_iterator ip;
//     for (ix = calmap.begin(); ix != calmap.end(); ++ix) {
//       int i = (*ix).first;
//       std::pair<int,int> p = channelToPixelROC(i);
//       it  = calmap.find(i);
//       CalParameters y  = (*it).second;
//       CalParameters z = (*ix).second;
//       cout << i <<" "<<p.first<<" "<<p.second<<" "<<y.p0<<" "<<z.p0<<" "<<calmap[i].p0<<endl;

//       //int dummy=0;
//       //cin>>dummy;
//     }

  } // end if readparameters
  return calmap;
} // end initCal()

//=========================================================================
SiPixelDigitizerAlgorithm::~SiPixelDigitizerAlgorithm() {
  LogDebug ("PixelDigitizer")<<"SiPixelDigitizerAlgorithm deleted";
}

SiPixelDigitizerAlgorithm::PixelEfficiencies::PixelEfficiencies(const edm::ParameterSet& conf, bool AddPixelInefficiency, int NumberOfBarrelLayers, int NumberOfEndcapDisks) {
  // pixel inefficiency
  // Don't use Hard coded values, read inefficiencies in from python or don't use any
  int NumberOfTotLayers = NumberOfBarrelLayers + NumberOfEndcapDisks;
  FPixIndex=NumberOfBarrelLayers;
  if (AddPixelInefficiency){
		     int i=0;
 	  	     thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix1");
 	  	     thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix2");
 	  	     thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix3");
		     if (NumberOfBarrelLayers>=4){thePixelColEfficiency[i++] = conf.getParameter<double>("thePixelColEfficiency_BPix4");}
		     //
		     i=0;
		     thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix1");
                     thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix2");
                     thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix3");
                     if (NumberOfBarrelLayers>=4){thePixelEfficiency[i++] = conf.getParameter<double>("thePixelEfficiency_BPix4");}
		     //
                     i=0;
		     thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix1");
                     thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix2");
                     thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix3");
                     if (NumberOfBarrelLayers>=4){thePixelChipEfficiency[i++] = conf.getParameter<double>("thePixelChipEfficiency_BPix4");}
		     // The next is needed for Phase2 Tracker studies
		     if (NumberOfBarrelLayers>=5){
			if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
			// For Phase2 tracker layers just set the outermost BPix inefficiency to 99.9%
			for (int j=5 ; j<=NumberOfBarrelLayers ; j++){
			    thePixelColEfficiency[j-1]=0.999;
			    thePixelEfficiency[j-1]=0.999;
			    thePixelChipEfficiency[j-1]=0.999;
			}
		     }
		     //
                     thePixelColEfficiency[FPixIndex] 	= conf.getParameter<double>("thePixelColEfficiency_FPix");
                     thePixelEfficiency[FPixIndex] 	= conf.getParameter<double>("thePixelEfficiency_FPix");
                     thePixelChipEfficiency[FPixIndex] 	= conf.getParameter<double>("thePixelChipEfficiency_FPix");
  }
  // the first "NumberOfBarrelLayers" settings [0],[1], ... , [NumberOfBarrelLayers-1] are for the barrel pixels
  // the next  "NumberOfEndcapDisks"  settings [NumberOfBarrelLayers],[NumberOfBarrelLayers+1], ... are for the endcaps (undecided how)
  if(!AddPixelInefficiency) {  // No inefficiency, all 100% efficient
    for (int i=0; i<NumberOfTotLayers;i++) {
      thePixelEfficiency[i]     = 1.;  // pixels = 100%
      thePixelColEfficiency[i]  = 1.;  // columns = 100%
      thePixelChipEfficiency[i] = 1.;  // chips = 100%
    }
  }
}

//=========================================================================
void SiPixelDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                                  std::vector<PSimHit>::const_iterator inputEnd,
                                                  const PixelGeomDetUnit* pixdet,
                                                  const GlobalVector& bfield) {
    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits

    uint32_t detId = pixdet->geographicalId().rawId();
    for (std::vector<PSimHit>::const_iterator ssbegin = inputBegin; ssbegin != inputEnd; ++ssbegin) {
      // skip hits not in this detector.
      if((*ssbegin).detUnitId() != detId) {
        continue;
      }

#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer")
	<< (*ssbegin).particleType() << " " << (*ssbegin).pabs() << " "
	<< (*ssbegin).energyLoss() << " " << (*ssbegin).tof() << " "
	<< (*ssbegin).trackId() << " " << (*ssbegin).processType() << " "
	<< (*ssbegin).detUnitId()
	<< (*ssbegin).entryPoint() << " " << (*ssbegin).exitPoint() ;
#endif

      
      std::vector<EnergyDepositUnit> ionization_points;
      std::vector<SignalPoint> collection_points;

      // fill collection_points for this SimHit, indpendent of topology
      // Check the TOF cut
      //if(std::abs( (*ssbegin).tof() )<theTofCut) // old cut
      if( ((*ssbegin).tof() >= theTofLowerCut) && ((*ssbegin).tof() <= theTofUpperCut) ) {
	primary_ionization(*ssbegin, ionization_points); // fills _ionization_points
	drift(*ssbegin, pixdet, bfield, ionization_points, collection_points);  // transforms _ionization_points to collection_points
	// compute induced signal on readout elements and add to _signal
	induce_signal(*ssbegin, pixdet, collection_points); // *ihit needed only for SimHit<-->Digi link
      } //  end if
    } // end for

}

//============================================================================
void SiPixelDigitizerAlgorithm::digitize(const PixelGeomDetUnit* pixdet,
                                         std::vector<PixelDigi>& digis,
                                         std::vector<PixelDigiSimLink>& simlinks, const TrackerTopology *tTopo) {

   // Pixel Efficiency moved from the constructor to this method because
   // the information of the det are not available in the constructor
   // Effciency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi

   uint32_t detID = pixdet->geographicalId().rawId();
   const signal_map_type& theSignal = _signal[detID];

   const PixelTopology* topol=&pixdet->specificTopology();
   int numColumns = topol->ncolumns();  // det module number of cols&rows
   int numRows = topol->nrows();

    // Noise already defined in electrons
    //thePixelThresholdInE = thePixelThreshold * theNoiseInElectrons ;
    // Find the threshold in noise units, needed for the noiser.

  unsigned int Sub_detid=DetId(detID).subdetId();

  float thePixelThresholdInE = 0.;

  if(theNoiseInElectrons>0.){
    if(Sub_detid == PixelSubdetector::PixelBarrel){ // Barrel modules
      int lay = tTopo->pxbLayer(detID);
      if(addThresholdSmearing) {
	if(lay==1) {
	  thePixelThresholdInE = smearedThreshold_BPix_L1_->fire(); // gaussian smearing
	} else {
	  thePixelThresholdInE = smearedThreshold_BPix_->fire(); // gaussian smearing
	}
      } else {
	if(lay==1) {
	  thePixelThresholdInE = theThresholdInE_BPix_L1;
	} else {
	  thePixelThresholdInE = theThresholdInE_BPix; // no smearing
	}
      }
    } else { // Forward disks modules
      if(addThresholdSmearing) {
	thePixelThresholdInE = smearedThreshold_FPix_->fire(); // gaussian smearing
      } else {
	thePixelThresholdInE = theThresholdInE_FPix; // no smearing
      }
    }
  }


#ifdef TP_DEBUG
    // full detector thickness
   float moduleThickness = pixdet->specificSurface().bounds().thickness();
    LogDebug ("PixelDigitizer")
      << " PixelDigitizer "
      << numColumns << " " << numRows << " " << moduleThickness;
#endif

    if(addNoise) add_noise(pixdet, thePixelThresholdInE/theNoiseInElectrons);  // generate noise

    // Do only if needed

    if((AddPixelInefficiency) && (theSignal.size()>0))
      pixel_inefficiency(pixelEfficiencies_, pixdet, tTopo); // Kill some pixels

    if(use_ineff_from_db_ && (theSignal.size()>0))
      pixel_inefficiency_db(detID);

    if(use_module_killing_) {
      if (use_deadmodule_DB_) {  // remove dead modules using DB
        module_killing_DB(detID);
      } else { // remove dead modules using the list in cfg file
        module_killing_conf(detID);
      }
    }

    make_digis(thePixelThresholdInE, detID, digis, simlinks, tTopo);

#ifdef TP_DEBUG
  LogDebug ("PixelDigitizer") << "[SiPixelDigitizerAlgorithm] converted " << digis.size() << " PixelDigis in DetUnit" << detID;
#endif
}

//***********************************************************************/
// Generate primary ionization along the track segment.
// Divide the track into small sub-segments
void SiPixelDigitizerAlgorithm::primary_ionization(const PSimHit& hit, std::vector<EnergyDepositUnit>& ionization_points) const {

// Straight line approximation for trajectory inside active media

  const float SegmentLength = 0.0010; //10microns in cm
  float energy;

  // Get the 3D segment direction vector
  LocalVector direction = hit.exitPoint() - hit.entryPoint();

  float eLoss = hit.energyLoss();  // Eloss in GeV
  float length = direction.mag();  // Track length in Silicon

  int NumberOfSegments = int ( length / SegmentLength); // Number of segments
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

  ionization_points.resize( NumberOfSegments); // set size

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
    ionization_points[i] = edu; // save

#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer")
      << i << " " << ionization_points[i].x() << " "
      << ionization_points[i].y() << " "
      << ionization_points[i].z() << " "
      << ionization_points[i].energy();
#endif

  }  // end for loop

  delete[] elossVector;

}
//******************************************************************************

// Fluctuate the charge comming from a small (10um) track segment.
// Use the G4 routine. For mip pions for the moment.
void SiPixelDigitizerAlgorithm::fluctuateEloss(int pid, float particleMomentum,
				      float eloss, float length,
				      int NumberOfSegs,float elossVector[]) const {

  // Get dedx for this track
  //float dedx;
  //if( length > 0.) dedx = eloss/length;
  //else dedx = eloss;

  double particleMass = 139.6; // Mass in MeV, Assume pion
  pid = std::abs(pid);
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
void SiPixelDigitizerAlgorithm::drift(const PSimHit& hit,
			              const PixelGeomDetUnit* pixdet,
                                      const GlobalVector& bfield,
                                      const std::vector<EnergyDepositUnit>& ionization_points,
                                      std::vector<SignalPoint>& collection_points) const {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter drift " ;
#endif

  collection_points.resize(ionization_points.size()); // set size

  LocalVector driftDir=DriftDirection(pixdet, bfield, hit.detUnitId());  // get the charge drift direction
  if(driftDir.z() ==0.) {
    LogWarning("Magnetic field") << " pxlx: drift in z is zero ";
    return;
  }
  
  // tangent of Lorentz angle
  //float TanLorenzAngleX = driftDir.x()/driftDir.z();
  //float TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();
  
  float TanLorenzAngleX, TanLorenzAngleY,dir_z, CosLorenzAngleX,
    CosLorenzAngleY;
  if( alpha2Order) {

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

  float moduleThickness = pixdet->specificSurface().bounds().thickness();
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


  for (unsigned int i = 0; i != ionization_points.size(); i++) {

    float SegX, SegY, SegZ; // position
    SegX = ionization_points[i].x();
    SegY = ionization_points[i].y();
    SegZ = ionization_points[i].z();

    // Distance from the collection plane
    //DriftDistance = (moduleThickness/2. + SegZ); // Drift to -z
    // Include explixitely the E drift direction (for CMS dir_z=-1)
    DriftDistance = moduleThickness/2. - (dir_z * SegZ); // Drift to -z

    //if( DriftDistance <= 0.)
    //cout<<" <=0 "<<DriftDistance<<" "<<i<<" "<<SegZ<<" "<<dir_z<<" "
    //  <<SegX<<" "<<SegY<<" "<<(moduleThickness/2)<<" "
    //  <<ionization_points[i].energy()<<" "
    //  <<hit.particleType()<<" "<<hit.pabs()<<" "<<hit.energyLoss()<<" "
    //  <<hit.entryPoint()<<" "<<hit.exitPoint()
    //  <<std::endl;

    if( DriftDistance < 0.) {
      DriftDistance = 0.;
    } else if( DriftDistance > moduleThickness )
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
     Sigma_x, Sigma_y, hit.tof(), ionization_points[i].energy() );

    // Load the Charge distribution parameters
    collection_points[i] = (sp);

  } // loop over ionization points, i.

} // end drift

//*************************************************************************
// Induce the signal on the collection plane of the active sensor area.
void SiPixelDigitizerAlgorithm::induce_signal(const PSimHit& hit,
			                      const PixelGeomDetUnit* pixdet,
                                              const std::vector<SignalPoint>& collection_points) {

  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)

   const PixelTopology* topol=&pixdet->specificTopology();
   uint32_t detID= pixdet->geographicalId().rawId();
   signal_map_type& theSignal = _signal[detID];

#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer")
      << " enter induce_signal, "
      << topol->pitch().first << " " << topol->pitch().second; //OK
#endif

   // local map to store pixels hit by 1 Hit.
   typedef std::map< int, float, std::less<int> > hit_map_type;
   hit_map_type hit_signal;

   // map to store pixel integrals in the x and in the y directions
   std::map<int, float, std::less<int> > x,y;

   // Assign signals to readout channels and store sorted by channel number

   // Iterate over collection points on the collection plane
   for ( std::vector<SignalPoint>::const_iterator i=collection_points.begin();
	 i != collection_points.end(); ++i) {

     float CloudCenterX = i->position().x(); // Charge position in x
     float CloudCenterY = i->position().y(); //                 in y
     float SigmaX = i->sigma_x();            // Charge spread in x
     float SigmaY = i->sigma_y();            //               in y
     float Charge = i->amplitude();          // Charge amplitude


     //if(SigmaX==0 || SigmaY==0) {
     //cout<<SigmaX<<" "<<SigmaY
     //   << " cloud " << i->position().x() << " " << i->position().y() << " "
     //   << i->sigma_x() << " " << i->sigma_y() << " " << i->amplitude()<<std::endl;
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
     int numColumns = topol->ncolumns();  // det module number of cols&rows
     int numRows = topol->nrows();

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
       //if(ix == 0) LowerBound = 0.;
       if(ix == 0 || SigmaX==0. )  // skip for surface segemnts
	 LowerBound = 0.;
       else {
	 mp = MeasurementPoint( float(ix), 0.0);
	 xLB = topol->localPosition(mp).x();
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xLB-CloudCenterX)/SigmaX, &result);
	 if(status != 0)
	   LogWarning ("Integration")<<"could not compute gaussian probability";
	 LowerBound = 1-result.val;
       }

       if(ix == numRows-1 || SigmaX==0. )
	 UpperBound = 1.;
       else {
	 mp = MeasurementPoint( float(ix+1), 0.0);
	 xUB = topol->localPosition(mp).x();
	 gsl_sf_result result;
	 int status = gsl_sf_erf_Q_e( (xUB-CloudCenterX)/SigmaX, &result);
	 if(status != 0)
	   LogWarning ("Integration")<<"could not compute gaussian probability";
	 UpperBound = 1. - result.val;
       }

       float   TotalIntegrationRange = UpperBound - LowerBound; // get strip
       x[ix] = TotalIntegrationRange; // save strip integral
       //if(SigmaX==0 || SigmaY==0)
       //cout<<TotalIntegrationRange<<" "<<ix<<std::endl;

     }

    // Now integarte strips in y
    int iy; // TT for compatibility
    for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind
      float yUB, yLB, UpperBound, LowerBound;

      if(iy == 0 || SigmaY==0.)
	LowerBound = 0.;
      else {
        mp = MeasurementPoint( 0.0, float(iy) );
        yLB = topol->localPosition(mp).y();
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yLB-CloudCenterY)/SigmaY, &result);
	if(status != 0)
	  LogWarning ("Integration")<<"could not compute gaussian probability";
	LowerBound = 1. - result.val;
      }

      if(iy == numColumns-1 || SigmaY==0. )
	UpperBound = 1.;
      else {
        mp = MeasurementPoint( 0.0, float(iy+1) );
        yUB = topol->localPosition(mp).y();
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e( (yUB-CloudCenterY)/SigmaY, &result);
	if(status != 0)
	  LogWarning ("Integration")<<"could not compute gaussian probability";
	UpperBound = 1. - result.val;
      }

      float   TotalIntegrationRange = UpperBound - LowerBound;
      y[iy] = TotalIntegrationRange; // save strip integral
      //if(SigmaX==0 || SigmaY==0)
      //cout<<TotalIntegrationRange<<" "<<iy<<std::endl;
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
    //     std::pair<float,float> p = topol->pixel( i->position() );  //OK
    //     chan = PixelDigi::pixelToChannel( int(p.first), int(p.second));
    //     std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
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
	im != hit_signal.end(); ++im) {
    int chan =  (*im).first;
    theSignal[chan] += (makeDigiSimLinks_ ? Amplitude( (*im).second, &hit, (*im).second) : Amplitude( (*im).second, (*im).second) )  ;


#ifdef TP_DEBUG
    std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
    LogDebug ("Pixel Digitizer")
      << " pixel " << ip.first << " " << ip.second << " "
      << theSignal[chan];
#endif
  }

} // end induce_signal

/***********************************************************************/

// Build pixels, check threshold, add misscalibration, ...
void SiPixelDigitizerAlgorithm::make_digis(float thePixelThresholdInE,
                                           uint32_t detID,
                                           std::vector<PixelDigi>& digis,
                                           std::vector<PixelDigiSimLink>& simlinks,
					   const TrackerTopology *tTopo) const  {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " make digis "<<" "
			       << " pixel threshold FPix" << theThresholdInE_FPix << " "
                               << " pixel threshold BPix" << theThresholdInE_BPix << " "
                               << " pixel threshold BPix Layer1" << theThresholdInE_BPix_L1 << " "
			       << " List pixels passing threshold ";
#endif

  // Loop over hit pixels

  signalMaps::const_iterator it = _signal.find(detID);
  if (it == _signal.end()) {
    return;
  }

  const signal_map_type& theSignal = (*it).second;

  for (signal_map_const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {

    float signalInElectrons = (*i).second ;   // signal in electrons

    // Do the miss calibration for calibration studies only.
    //if(doMissCalibrate) signalInElectrons = missCalibrate(signalInElectrons)

    // Do only for pixels above threshold

    if( signalInElectrons >= thePixelThresholdInE) { // check threshold

      int chan =  (*i).first;  // channel number
      std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
      int adc=0;  // ADC count as integer

      // Do the miss calibration for calibration studies only.
      if(doMissCalibrate) {
	int row = ip.first;  // X in row
	int col = ip.second; // Y is in col
	adc = int(missCalibrate(detID, col, row, signalInElectrons)); //full misscalib.
      } else { // Just do a simple electron->adc conversion
	adc = int( signalInElectrons / theElectronPerADC ); // calibrate gain
      }
      adc = std::min(adc, theAdcFullScale); // Check maximum value
// Calculate layerIndex
     if (theAdcFullScale!=theAdcFullScaleStack){
        unsigned int Sub_detid=DetId(detID).subdetId();
        if(Sub_detid == PixelSubdetector::PixelBarrel){ // Barrel modules
          int lay = tTopo->pxbLayer(detID);
          if (lay>=theFirstStackLayer) {
            // Set to 1 if over the threshold
            if (theAdcFullScaleStack==1) {adc=1;}
            // Make it a linear fit to the full scale of the normal adc count.   Start new adc from 1 not zero.
            if (theAdcFullScaleStack!=1&&theAdcFullScaleStack!=theAdcFullScale) {adc = int (1 + adc * (theAdcFullScaleStack-1)/float(theAdcFullScale) );}
          }
        }
     } // Only enter this if the Adc changes for the outer layers
#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer")
	<< (*i).first << " " << (*i).second << " " << signalInElectrons
	<< " " << adc << ip.first << " " << ip.second ;
#endif

      // Load digis
      digis.emplace_back(ip.first, ip.second, adc);

      if (makeDigiSimLinks_) {
        //digilink
        if((*i).second.trackIds().size()>0){
          simlink_map simi;
	  unsigned int il=0;
	  for( std::vector<unsigned int>::const_iterator itid = (*i).second.trackIds().begin();
	       itid != (*i).second.trackIds().end(); ++itid) {
	    simi[*itid].push_back((*i).second.individualampl()[il]);
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
	    if(fraction>1.) fraction=1.;
	    simlinks.emplace_back((*i).first, (*simiiter).first, (*i).second.eventId(), fraction);
	  }
        }
      }
    }
  }
}

/***********************************************************************/

//  Add electronic noise to pixel charge
void SiPixelDigitizerAlgorithm::add_noise(const PixelGeomDetUnit* pixdet,
                                          float thePixelThreshold) {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter add_noise " << theNoiseInElectrons;
#endif

  uint32_t detID= pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];


  // First add noise to hit pixels
  float theSmearedChargeRMS = 0.0;

  for ( signal_map_iterator i = theSignal.begin(); i != theSignal.end(); i++) {

         if(addChargeVCALSmearing)
      {
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
	// Noise from full readout:
	float noise  = gaussDistribution_->fire() ;

		if(((*i).second + Amplitude(noise+noise_ChargeVCALSmearing, -1.)) < 0. ) {
		  (*i).second.set(0);}
		else{
	(*i).second +=Amplitude(noise+noise_ChargeVCALSmearing, -1.);
		}

      } // End if addChargeVCalSmearing
	 else
     {
	// Noise: ONLY full READOUT Noise.
	// Use here the FULL readout noise, including TBM,ALT,AOH,OPT-REC.

	float noise  = gaussDistribution_->fire() ;
		if(((*i).second + Amplitude(noise, -1.)) < 0. ) {
		  (*i).second.set(0);}
		else{
	(*i).second +=Amplitude(noise, -1.);
		}
     } // end if only Noise from full readout

  }

  if(!addNoisyPixels)  // Option to skip noise in non-hit pixels
    return;

  const PixelTopology* topol=&pixdet->specificTopology();
  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();

  // Add noise on non-hit pixels
  // Use here the pixel noise
  int numberOfPixels = (numRows * numColumns);
  std::map<int,float, std::less<int> > otherPixels;
  std::map<int,float, std::less<int> >::iterator mapI;

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

    if(theSignal[chan] == 0){
      //      float noise = float( (*mapI).second );
      int noise=int( (*mapI).second );
      theSignal[chan] = Amplitude (noise, -1.);
    }
  }
}

/***********************************************************************/

// Simulate the readout inefficiencies.
// Delete a selected number of single pixels, dcols and rocs.
void SiPixelDigitizerAlgorithm::pixel_inefficiency(const PixelEfficiencies& eff,
			                           const PixelGeomDetUnit* pixdet,
						   const TrackerTopology *tTopo) {

  uint32_t detID= pixdet->geographicalId().rawId();

  signal_map_type& theSignal = _signal[detID];

  const PixelTopology* topol=&pixdet->specificTopology();
  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();

  // Predefined efficiencies
  float pixelEfficiency  = 1.0;
  float columnEfficiency = 1.0;
  float chipEfficiency   = 1.0;

  // setup the chip indices conversion
  unsigned int Subid=DetId(detID).subdetId();
  if    (Subid==  PixelSubdetector::PixelBarrel){// barrel layers
    int layerIndex=tTopo->pxbLayer(detID);
    pixelEfficiency  = eff.thePixelEfficiency[layerIndex-1];
    columnEfficiency = eff.thePixelColEfficiency[layerIndex-1];
    chipEfficiency   = eff.thePixelChipEfficiency[layerIndex-1];

    // This should never happen, but only check if it is not an upgrade geometry
    if (NumberOfBarrelLayers==3){
       if(numColumns>416)  LogWarning ("Pixel Geometry") <<" wrong columns in barrel "<<numColumns;
       if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in barrel "<<numRows;
    }
  } else {                // forward disks

    // For endcaps take same for each endcap
    pixelEfficiency  = eff.thePixelEfficiency[eff.FPixIndex];
    columnEfficiency = eff.thePixelColEfficiency[eff.FPixIndex];
    chipEfficiency   = eff.thePixelChipEfficiency[eff.FPixIndex];
    // Sometimes the forward pixels have wrong size,
    // this crashes the index conversion, so exit, but only check if it is not an upgrade geometry
    if (NumberOfBarrelLayers==3){
       if(numColumns>260 || numRows>160) {
         if(numColumns>260)  LogWarning ("Pixel Geometry") <<" wrong columns in endcaps "<<numColumns;
         if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in endcaps "<<numRows;
         return;
       }
    }
  } // if barrel/forward

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter pixel_inefficiency " << pixelEfficiency << " "
			       << columnEfficiency << " " << chipEfficiency;
#endif

  // Initilize the index converter
  //PixelIndices indexConverter(numColumns,numRows);
  std::auto_ptr<PixelIndices> pIndexConverter(new PixelIndices(numColumns,numRows));

  int chipIndex = 0;
  int rowROC = 0;
  int colROC = 0;
  std::map<int, int, std::less<int> >chips, columns;
  std::map<int, int, std::less<int> >::iterator iter;

  // Find out the number of columns and rocs hits
  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for (signal_map_const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {

    int chan = i->first;
    std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
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
  for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {

    //    int chan = i->first;
    std::pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
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
float SiPixelDigitizerAlgorithm::missCalibrate(uint32_t detID, int col,int row,
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

    if(Sub_detid == PixelSubdetector::PixelBarrel){// barrel layers
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
  //std::auto_ptr<PixelIndices> pIndexConverter(new PixelIndices(numColumns,numRows));
  //pIndexConverter->transformToROC(col,row,chipIndex,colROC,rowROC);

  // Use calibration from a file
  //int chanROC = PixelIndices::pixelToChannelROC(rowROC,colROC); // use ROC coordinates
  //float pp0=0, pp1=0,pp2=0,pp3=0;
  //map<int,CalParameters,std::less<int> >::const_iterator it=calmap.find(chanROC);
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
  //  <<(signalInElectrons/theElectronPerADC)<<std::endl;

  return newAmp;
}
//******************************************************************************

// Set the drift direction accoring to the Bfield in local det-unit frame
// Works for both barrel and forward pixels.
// Replace the sign convention to fit M.Swartz's formulaes.
// Configurations for barrel and foward pixels possess different tanLorentzAngleperTesla
// parameter value

LocalVector SiPixelDigitizerAlgorithm::DriftDirection(const PixelGeomDetUnit* pixdet,
                                                      const GlobalVector& bfield,
                                                      const DetId& detId) const {
  Frame detFrame(pixdet->surface().position(),pixdet->surface().rotation());
  LocalVector Bfield=detFrame.toLocal(bfield);
  
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

  uint32_t detID= pixdet->geographicalId().rawId();

  unsigned int Sub_detid=DetId(detID).subdetId();

  // Read Lorentz angle from cfg file:**************************************************************

  if(!use_LorentzAngle_DB_){
    
    if( alpha2Order) {
      alpha2_FPix = tanLorentzAnglePerTesla_FPix*tanLorentzAnglePerTesla_FPix;
      alpha2_BPix = tanLorentzAnglePerTesla_BPix*tanLorentzAnglePerTesla_BPix;
    }else {
      alpha2_FPix = 0.0;
      alpha2_BPix = 0.0;
    }
    
    if(Sub_detid == PixelSubdetector::PixelBarrel){// barrel layers
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
    float lorentzAngle = SiPixelLorentzAngle_->getLorentzAngle(detId);
    alpha2 = lorentzAngle * lorentzAngle;
    //std::cout << "detID is: "<< it->first <<"The LA per tesla is: "<< it->second << std::std::endl;
    dir_x = -( lorentzAngle * Bfield.y() + alpha2 * Bfield.z()* Bfield.x() );
    dir_y = +( lorentzAngle * Bfield.x() - alpha2 * Bfield.z()* Bfield.y() );
    dir_z = -(1 + alpha2 * Bfield.z()*Bfield.z() );
    scale = (1 + alpha2 * Bfield.z()*Bfield.z() );
  }// end: Read LA from DataBase.
  
  LocalVector theDriftDirection = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );
  
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " The drift direction in local coordinate is "
			       << theDriftDirection ;
#endif
  
  return theDriftDirection;
}

//****************************************************************************************************

void SiPixelDigitizerAlgorithm::pixel_inefficiency_db(uint32_t detID) {
  if(!use_ineff_from_db_)
    return;
  
  signal_map_type& theSignal = _signal[detID];

  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {

    //    int chan = i->first;
    std::pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
    int row = ip.first;  // X in row
    int col = ip.second; // Y is in col
    //transform to ROC index coordinates
    if(theSiPixelGainCalibrationService_->isDead(detID, col, row)){
      //      std::cout << "now in isdead check, row " << detID << " " << col << "," << row << std::std::endl;
      // make pixel amplitude =0, pixel will be lost at clusterization
      i->second.set(0.); // reset amplitude,
    } // end if
  } // end pixel loop
} // end pixel_indefficiency


//****************************************************************************************************

void SiPixelDigitizerAlgorithm::module_killing_conf(uint32_t detID) {
  if(!use_module_killing_)
    return;
  
  bool isbad=false;
  
  Parameters::const_iterator itDeadModules=DeadModules.begin();
  
  int detid = detID;
  for(; itDeadModules != DeadModules.end(); ++itDeadModules){
    int Dead_detID = itDeadModules->getParameter<int>("Dead_detID");
    if(detid == Dead_detID){
      isbad=true;
      break;
    }
  }
  
  if(!isbad)
    return;

  signal_map_type& theSignal = _signal[detID];
  
  std::string Module = itDeadModules->getParameter<std::string>("Module");
  
  if(Module=="whole"){
    for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
      i->second.set(0.); // reset amplitude
    }
  }
  
  for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
    std::pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos

    if(Module=="tbmA" && ip.first>=80 && ip.first<=159){
      i->second.set(0.);
    }

    if( Module=="tbmB" && ip.first<=79){
      i->second.set(0.);
    }
  }
}
//****************************************************************************************************
void SiPixelDigitizerAlgorithm::module_killing_DB(uint32_t detID) {
// Not SLHC safe for now
  if(!use_module_killing_)
    return;
  
  bool isbad=false;
  
  std::vector<SiPixelQuality::disabledModuleType>disabledModules = SiPixelBadModule_->getBadComponentList();
  
  SiPixelQuality::disabledModuleType badmodule;
  
  for (size_t id=0;id<disabledModules.size();id++)
    {
      if(detID==disabledModules[id].DetID){
	isbad=true;
        badmodule = disabledModules[id];
	break;
      }
    }
  
  if(!isbad)
    return;

  signal_map_type& theSignal = _signal[detID];
  
  //std::cout<<"Hit in: "<< detID <<" errorType "<< badmodule.errorType<<" BadRocs="<<std::hex<<SiPixelBadModule_->getBadRocs(detID)<<dec<<" "<<std::endl;
  if(badmodule.errorType == 0){ // this is a whole dead module.
    
    for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
      i->second.set(0.); // reset amplitude
    }
  }
  else { // all other module types: half-modules and single ROCs.
    // Get Bad ROC position:
    //follow the example of getBadRocPositions in CondFormats/SiPixelObjects/src/SiPixelQuality.cc
    std::vector<GlobalPixel> badrocpositions (0);
    for(unsigned int j = 0; j < 16; j++){
      if(SiPixelBadModule_->IsRocBad(detID, j) == true){
	
	std::vector<CablingPathToDetUnit> path = map_.product()->pathToDetUnit(detID);
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
    
    
    for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
      std::pair<int,int> ip = PixelDigi::channelToPixel(i->first);//get pixel pos
      
      for(std::vector<GlobalPixel>::const_iterator it = badrocpositions.begin(); it != badrocpositions.end(); ++it){
	if(it->row >= 80 && ip.first >= 80 ){
	  if((fabs(ip.second - it->col) < 26) ) {i->second.set(0.);}
          else if(it->row==120 && ip.second-it->col==26){i->second.set(0.);}
          else if(it->row==119 && it->col-ip.second==26){i->second.set(0.);}
	}
	else if(it->row < 80 && ip.first < 80 ){
	  if((fabs(ip.second - it->col) < 26) ){i->second.set(0.);}
          else if(it->row==40 && ip.second-it->col==26){i->second.set(0.);}
          else if(it->row==39 && it->col-ip.second==26){i->second.set(0.);}
       }
      }
    }
  }
}

