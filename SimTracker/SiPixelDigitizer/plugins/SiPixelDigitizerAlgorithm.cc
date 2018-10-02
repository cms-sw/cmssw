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
// February, 2018: Implement cluster charge reweighting (P. Schuetze, with code from A. Hazi)
#include <iostream>
#include <iomanip>

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
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"

#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"

#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h" 

#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"



using namespace edm;
using namespace sipixelobjects;


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

  // Read template files for charge reweighting
  if (UseReweighting){
    edm::ESHandle<SiPixel2DTemplateDBObject> SiPixel2DTemp_den;
    es.get<SiPixel2DTemplateDBObjectRcd>().get("denominator",SiPixel2DTemp_den);
    dbobject_den = SiPixel2DTemp_den.product();
    
    edm::ESHandle<SiPixel2DTemplateDBObject> SiPixel2DTemp_num;
    es.get<SiPixel2DTemplateDBObjectRcd>().get("numerator",SiPixel2DTemp_num);
    dbobject_num = SiPixel2DTemp_num.product();
    
    int numOfTemplates = dbobject_den->numOfTempl()+dbobject_num->numOfTempl();
    templateStores_.reserve(numOfTemplates);
    SiPixelTemplate2D::pushfile(*dbobject_den, templateStores_);
    SiPixelTemplate2D::pushfile(*dbobject_num, templateStores_);
    
    track.resize(6);
  }
}

//=========================================================================

SiPixelDigitizerAlgorithm::SiPixelDigitizerAlgorithm(const edm::ParameterSet& conf) :

  _signal(),
  makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", true)),
  use_ineff_from_db_(conf.getParameter<bool>("useDB")),
  use_module_killing_(conf.getParameter<bool>("killModules")), // boolean to kill or not modules
  use_deadmodule_DB_(conf.getParameter<bool>("DeadModules_DB")), // boolean to access dead modules from DB
  use_LorentzAngle_DB_(conf.getParameter<bool>("LorentzAngle_DB")), // boolean to access Lorentz angle from DB

  DeadModules(use_deadmodule_DB_ ? Parameters() : conf.getParameter<Parameters>("DeadModules")), // get dead module from cfg file

  templ2D(templateStores_),
  xdouble(TXSIZE),
  ydouble(TYSIZE),
  IDnum(conf.exists("TemplateIDnumerator")?conf.getParameter<int>("TemplateIDnumerator"):0),
  IDden(conf.exists("TemplateIDdenominator")?conf.getParameter<int>("TemplateIDdenominator"):0),

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
  theThresholdInE_BPix_L2(conf.exists("ThresholdInElectrons_BPix_L2")?conf.getParameter<double>("ThresholdInElectrons_BPix_L2"):theThresholdInE_BPix),

  // Add threshold gaussian smearing:
  theThresholdSmearing_FPix(conf.getParameter<double>("ThresholdSmearing_FPix")),
  theThresholdSmearing_BPix(conf.getParameter<double>("ThresholdSmearing_BPix")),
  theThresholdSmearing_BPix_L1(conf.exists("ThresholdSmearing_BPix_L1")?conf.getParameter<double>("ThresholdSmearing_BPix_L1"):theThresholdSmearing_BPix),
  theThresholdSmearing_BPix_L2(conf.exists("ThresholdSmearing_BPix_L2")?conf.getParameter<double>("ThresholdSmearing_BPix_L2"):theThresholdSmearing_BPix),

  // electrons to VCAL conversion needed in misscalibrate()
  electronsPerVCAL(conf.getParameter<double>("ElectronsPerVcal")),
  electronsPerVCAL_Offset(conf.getParameter<double>("ElectronsPerVcal_Offset")),
  electronsPerVCAL_L1(conf.exists("ElectronsPerVcal_L1")?conf.getParameter<double>("ElectronsPerVcal_L1"):electronsPerVCAL),
  electronsPerVCAL_L1_Offset(conf.exists("ElectronsPerVcal_L1_Offset")?conf.getParameter<double>("ElectronsPerVcal_L1_Offset"):electronsPerVCAL_Offset),

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
  AddPixelInefficiency(conf.getParameter<bool>("AddPixelInefficiency")),

  // Add threshold gaussian smearing:
  addThresholdSmearing(conf.getParameter<bool>("AddThresholdSmearing")),

  // Get the constants for the miss-calibration studies
  doMissCalibrate(conf.getParameter<bool>("MissCalibrate")), // Enable miss-calibration
  theGainSmearing(conf.getParameter<double>("GainSmearing")), // sigma of the gain smearing
  theOffsetSmearing(conf.getParameter<double>("OffsetSmearing")), //sigma of the offset smearing

  // Add pixel radiation damage for upgrade studies
  AddPixelAging(conf.getParameter<bool>("DoPixelAging")),
  UseReweighting(conf.getParameter<bool>("UseReweighting")),
  PrintClusters(conf.getParameter<bool>("PrintClusters")),
  PrintTemplates(conf.getParameter<bool>("PrintTemplates")),

  // delta cutoff in MeV, has to be same as in OSCAR(0.030/cmsim=1.0 MeV
  //tMax(0.030), // In MeV.
  //tMax(conf.getUntrackedParameter<double>("deltaProductionCut",0.030)),
  tMax(conf.getParameter<double>("deltaProductionCut")),

  fluctuate(fluctuateCharge ? new SiG4UniversalFluctuation() : nullptr),
  theNoiser(addNoise ? new GaussianTailNoiseGenerator() : nullptr),
  calmap(doMissCalibrate ? initCal() : std::map<int,CalParameters,std::less<int> >()),
  theSiPixelGainCalibrationService_(use_ineff_from_db_ ? new SiPixelGainCalibrationOfflineSimService(conf) : nullptr),
  pixelEfficiencies_(conf, AddPixelInefficiency,NumberOfBarrelLayers,NumberOfEndcapDisks),
  pixelAging_(conf,AddPixelAging,NumberOfBarrelLayers,NumberOfEndcapDisks)
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
                              << "threshold in electron BPix Layer2 = "
                              << theThresholdInE_BPix_L2
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

// Read DynIneff Scale factors from Configuration
SiPixelDigitizerAlgorithm::PixelEfficiencies::PixelEfficiencies(const edm::ParameterSet& conf, bool AddPixelInefficiency, int NumberOfBarrelLayers, int NumberOfEndcapDisks) {
  // pixel inefficiency
  // Don't use Hard coded values, read inefficiencies in from DB/python config or don't use any
  int NumberOfTotLayers = NumberOfBarrelLayers + NumberOfEndcapDisks;
  FPixIndex=NumberOfBarrelLayers;
  if (AddPixelInefficiency){
    FromConfig = 
      conf.exists("thePixelColEfficiency_BPix1") && conf.exists("thePixelColEfficiency_BPix2") && conf.exists("thePixelColEfficiency_BPix3") &&
      conf.exists("thePixelColEfficiency_FPix1") && conf.exists("thePixelColEfficiency_FPix2") &&
      conf.exists("thePixelEfficiency_BPix1") && conf.exists("thePixelEfficiency_BPix2") && conf.exists("thePixelEfficiency_BPix3") &&
      conf.exists("thePixelEfficiency_FPix1") && conf.exists("thePixelEfficiency_FPix2") &&
      conf.exists("thePixelChipEfficiency_BPix1") && conf.exists("thePixelChipEfficiency_BPix2") && conf.exists("thePixelChipEfficiency_BPix3") &&
      conf.exists("thePixelChipEfficiency_FPix1") && conf.exists("thePixelChipEfficiency_FPix2");
    if (NumberOfBarrelLayers==3) FromConfig = FromConfig && conf.exists("theLadderEfficiency_BPix1") && conf.exists("theLadderEfficiency_BPix2") && conf.exists("theLadderEfficiency_BPix3") &&
      conf.exists("theModuleEfficiency_BPix1") && conf.exists("theModuleEfficiency_BPix2") && conf.exists("theModuleEfficiency_BPix3") &&
      conf.exists("thePUEfficiency_BPix1") && conf.exists("thePUEfficiency_BPix2") && conf.exists("thePUEfficiency_BPix3") &&
      conf.exists("theInnerEfficiency_FPix1") && conf.exists("theInnerEfficiency_FPix2") &&
      conf.exists("theOuterEfficiency_FPix1") && conf.exists("theOuterEfficiency_FPix2") &&
      conf.exists("thePUEfficiency_FPix_Inner") && conf.exists("thePUEfficiency_FPix_Outer") &&
      conf.exists("theInstLumiScaleFactor");
    if (NumberOfBarrelLayers>=4) FromConfig = FromConfig && conf.exists("thePixelColEfficiency_BPix4") &&
      conf.exists("thePixelEfficiency_BPix4") && conf.exists("thePixelChipEfficiency_BPix4");
    if (NumberOfEndcapDisks>=3) FromConfig = FromConfig && conf.exists("thePixelColEfficiency_FPix4") &&
      conf.exists("thePixelEfficiency_FPix3") && conf.exists("thePixelChipEfficiency_FPix3");
    if (FromConfig) {
      LogInfo ("PixelDigitizer ") <<"The PixelDigitizer inefficiency configuration is read from the config file.\n";
      theInstLumiScaleFactor = conf.getParameter<double>("theInstLumiScaleFactor");
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
      //
      if (NumberOfBarrelLayers==3){
        i=0;
        theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix1");
        theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix2");
        theLadderEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theLadderEfficiency_BPix3");
        if ( ((theLadderEfficiency_BPix[0].size()!=20) || (theLadderEfficiency_BPix[1].size()!=32) ||
              (theLadderEfficiency_BPix[2].size()!=44)) && (NumberOfBarrelLayers==3) )  
          throw cms::Exception("Configuration") << "Wrong ladder number in efficiency config!";
        //		     
        i=0;
        theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix1");
        theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix2");
        theModuleEfficiency_BPix[i++] = conf.getParameter<std::vector<double> >("theModuleEfficiency_BPix3");
        if ( ((theModuleEfficiency_BPix[0].size()!=4) || (theModuleEfficiency_BPix[1].size()!=4) ||
              (theModuleEfficiency_BPix[2].size()!=4)) && (NumberOfBarrelLayers==3) )  
          throw cms::Exception("Configuration") << "Wrong module number in efficiency config!";
        //
        thePUEfficiency.push_back(conf.getParameter<std::vector<double> >("thePUEfficiency_BPix1"));
        thePUEfficiency.push_back(conf.getParameter<std::vector<double> >("thePUEfficiency_BPix2"));
        thePUEfficiency.push_back(conf.getParameter<std::vector<double> >("thePUEfficiency_BPix3"));		    		    
        if ( ((thePUEfficiency[0].empty()) || (thePUEfficiency[1].empty()) || 
              (thePUEfficiency[2].empty())) && (NumberOfBarrelLayers==3) )
          throw cms::Exception("Configuration") << "At least one PU efficiency (BPix) number is needed in efficiency config!";
      }
      // The next is needed for Phase2 Tracker studies
      if (NumberOfBarrelLayers>=5){
        if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
        // For Phase2 tracker layers just set the outermost BPix inefficiency to 99.9% THESE VALUES ARE HARDCODED ALSO ELSEWHERE IN THIS FILE
        for (int j=5 ; j<=NumberOfBarrelLayers ; j++){
          thePixelColEfficiency[j-1]=0.999;
          thePixelEfficiency[j-1]=0.999;
          thePixelChipEfficiency[j-1]=0.999;
        }
      }
      //
      i=FPixIndex;
      thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix1");
      thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelColEfficiency[i++]   = conf.getParameter<double>("thePixelColEfficiency_FPix3");}
      i=FPixIndex;
      thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix1");
      thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelEfficiency[i++]      = conf.getParameter<double>("thePixelEfficiency_FPix3");}
      i=FPixIndex;
      thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix1");
      thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix2");
      if (NumberOfEndcapDisks>=3){thePixelChipEfficiency[i++]  = conf.getParameter<double>("thePixelChipEfficiency_FPix3");}
      // The next is needed for Phase2 Tracker studies
      if (NumberOfEndcapDisks>=4){
        if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
        // For Phase2 tracker layers just set the extra FPix disk inefficiency to 99.9% THESE VALUES ARE HARDCODED ALSO ELSEWHERE IN THIS FILE
        for (int j=4+FPixIndex ; j<=NumberOfEndcapDisks+NumberOfBarrelLayers ; j++){
          thePixelColEfficiency[j-1]=0.999;
          thePixelEfficiency[j-1]=0.999;
          thePixelChipEfficiency[j-1]=0.999;
        }
      }
      //FPix Dynamic Inefficiency
      if (NumberOfBarrelLayers==3){
        i=FPixIndex;
        theInnerEfficiency_FPix[i++] = conf.getParameter<double>("theInnerEfficiency_FPix1");
        theInnerEfficiency_FPix[i++] = conf.getParameter<double>("theInnerEfficiency_FPix2");
        i=FPixIndex;
        theOuterEfficiency_FPix[i++] = conf.getParameter<double>("theOuterEfficiency_FPix1");
        theOuterEfficiency_FPix[i++] = conf.getParameter<double>("theOuterEfficiency_FPix2");
        thePUEfficiency.push_back(conf.getParameter<std::vector<double> >("thePUEfficiency_FPix_Inner"));
        thePUEfficiency.push_back(conf.getParameter<std::vector<double> >("thePUEfficiency_FPix_Outer"));
        if ( ((thePUEfficiency[3].empty()) || (thePUEfficiency[4].empty())) && (NumberOfEndcapDisks==2) )
          throw cms::Exception("Configuration") << "At least one (FPix) PU efficiency number is needed in efficiency config!";
        pu_scale.resize(thePUEfficiency.size());
      }
    }
    else LogInfo ("PixelDigitizer ") <<"The PixelDigitizer inefficiency configuration is read from the database.\n";
  }
  // the first "NumberOfBarrelLayers" settings [0],[1], ... , [NumberOfBarrelLayers-1] are for the barrel pixels
  // the next  "NumberOfEndcapDisks"  settings [NumberOfBarrelLayers],[NumberOfBarrelLayers+1], ... [NumberOfEndcapDisks+NumberOfBarrelLayers-1]
}

// Read DynIneff Scale factors from DB
void SiPixelDigitizerAlgorithm::init_DynIneffDB(const edm::EventSetup& es, const unsigned int& bunchspace){
  if (AddPixelInefficiency&&!pixelEfficiencies_.FromConfig) {
    if (bunchspace == 50) es.get<SiPixelDynamicInefficiencyRcd>().get("50ns",SiPixelDynamicInefficiency_);
    else es.get<SiPixelDynamicInefficiencyRcd>().get(SiPixelDynamicInefficiency_);
    pixelEfficiencies_.init_from_db(geom_, SiPixelDynamicInefficiency_);
  }
}

void SiPixelDigitizerAlgorithm::PixelEfficiencies::init_from_db(const edm::ESHandle<TrackerGeometry>& geom, const edm::ESHandle<SiPixelDynamicInefficiency>& SiPixelDynamicInefficiency) {
  
  theInstLumiScaleFactor = SiPixelDynamicInefficiency->gettheInstLumiScaleFactor();
  const std::map<uint32_t, double>& PixelGeomFactorsDBIn = SiPixelDynamicInefficiency->getPixelGeomFactors();
  const std::map<uint32_t, double>& ColGeomFactorsDB = SiPixelDynamicInefficiency->getColGeomFactors();
  const std::map<uint32_t, double>& ChipGeomFactorsDB = SiPixelDynamicInefficiency->getChipGeomFactors();
  const std::map<uint32_t, std::vector<double> >& PUFactors = SiPixelDynamicInefficiency->getPUFactors();
  std::vector<uint32_t > DetIdmasks = SiPixelDynamicInefficiency->getDetIdmasks();
  
  // Loop on all modules, initialize map for easy access
  for( const auto& it_module : geom->detUnits()) {
    if( dynamic_cast<PixelGeomDetUnit const*>(it_module)==nullptr) continue;
    const DetId detid = it_module->geographicalId();
    uint32_t rawid = detid.rawId();
    PixelGeomFactors[rawid] = 1;
    ColGeomFactors[rawid] = 1;
    ChipGeomFactors[rawid] = 1;
    PixelGeomFactorsROCStdPixels[rawid] = std::vector<double>(16,1);
    PixelGeomFactorsROCBigPixels[rawid] = std::vector<double>(16,1);
  }
  
  // ROC level inefficiency for phase 1 (disentangle scale factors for big and std size pixels)  
  std::map<uint32_t, double>  PixelGeomFactorsDB;
  
  if (geom->isThere(GeomDetEnumerators::P1PXB)
      || geom->isThere(GeomDetEnumerators::P1PXEC)){
    for (auto db_factor : PixelGeomFactorsDBIn){  
      int shift = DetId(db_factor.first).subdetId() == 
	static_cast<int>(PixelSubdetector::PixelBarrel) ? BPixRocIdShift:FPixRocIdShift;          
      unsigned int rocMask = rocIdMaskBits << shift;
      unsigned int rocId = (((db_factor.first) & rocMask) >> shift);
      if (rocId != 0) {
	rocId--;
	unsigned int rawid = db_factor.first & (~rocMask);
	const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*> (geom->idToDet(rawid));	
	PixelTopology const * topology = &(theGeomDet->specificTopology());      
	const int nPixelsInROC = topology->rowsperroc()*topology->colsperroc();
	const int nBigPixelsInROC =  2*topology->rowsperroc()+topology->colsperroc()-2;
	double factor = db_factor.second;
	double badFraction = 1 - factor;
	double bigPixelFraction = static_cast<double> (nBigPixelsInROC)/nPixelsInROC;      
	double stdPixelFraction = 1. - bigPixelFraction;      
	
	double badFractionBig = std::min(bigPixelFraction, badFraction);  
	double badFractionStd = std::max(0., badFraction - badFractionBig);
	double badFractionBigReNormalized = badFractionBig/bigPixelFraction;
	double badFractionStdReNormalized = badFractionStd/stdPixelFraction;
	PixelGeomFactorsROCStdPixels[rawid][rocId] *= (1. - badFractionStdReNormalized);
	PixelGeomFactorsROCBigPixels[rawid][rocId] *= (1. - badFractionBigReNormalized);     
      }
      else{
	PixelGeomFactorsDB[db_factor.first] = db_factor.second;      
      }
    } 
  } // is Phase 1 geometry
  else{
    PixelGeomFactorsDB = PixelGeomFactorsDBIn;
  }
  
  // Loop on all modules, store module level geometrical scale factors 
  for( const auto& it_module : geom->detUnits()) {
    if( dynamic_cast<PixelGeomDetUnit const*>(it_module)==nullptr) continue;
    const DetId detid = it_module->geographicalId();
    uint32_t rawid = detid.rawId();
    for (auto db_factor : PixelGeomFactorsDB) if (matches(detid, DetId(db_factor.first), DetIdmasks)) PixelGeomFactors[rawid] *= db_factor.second;
    for (auto db_factor : ColGeomFactorsDB) if (matches(detid, DetId(db_factor.first), DetIdmasks)) ColGeomFactors[rawid] *= db_factor.second;
    for (auto db_factor : ChipGeomFactorsDB) if (matches(detid, DetId(db_factor.first), DetIdmasks)) ChipGeomFactors[rawid] *= db_factor.second;
  }
  
  // piluep scale factors are calculated once per event
  // therefore vector index is stored in a map for each module that matches to a db_id
  size_t i=0;
  for (auto factor : PUFactors) {
    const DetId db_id = DetId(factor.first);
    for( const auto& it_module : geom->detUnits()) {
      if( dynamic_cast<PixelGeomDetUnit const*>(it_module)==nullptr) continue;
      const DetId detid = it_module->geographicalId();
      if (!matches(detid, db_id, DetIdmasks)) continue;
      if (iPU.count(detid.rawId())) {
	throw cms::Exception("Database")<<"Multiple db_ids match to same module in SiPixelDynamicInefficiency DB Object";
      } else {
	iPU[detid.rawId()] = i;
      }
    }
    thePUEfficiency.push_back(factor.second);
    ++i;
  }
  pu_scale.resize(thePUEfficiency.size());
}

bool SiPixelDigitizerAlgorithm::PixelEfficiencies::matches(const DetId& detid, const DetId& db_id, const std::vector<uint32_t >& DetIdmasks) {
  if (detid.subdetId() != db_id.subdetId()) return false;
  for (size_t i=0; i<DetIdmasks.size(); ++i) {
    DetId maskid = DetId(DetIdmasks.at(i));
    if (maskid.subdetId() != db_id.subdetId()) continue;
    if ((detid.rawId()&maskid.rawId()) != (db_id.rawId()&maskid.rawId()) && 
	(db_id.rawId()&maskid.rawId()) != DetId(db_id.det(), db_id.subdetId()).rawId()) return false;
  }
  return true;
}

SiPixelDigitizerAlgorithm::PixelAging::PixelAging(const edm::ParameterSet& conf, bool AddAging, int NumberOfBarrelLayers, int NumberOfEndcapDisks) {
  // pixel aging
  // Don't use Hard coded values, read aging in from python or don't use any
  if(AddAging) {
    int NumberOfTotLayers = NumberOfBarrelLayers + NumberOfEndcapDisks;
    FPixIndex=NumberOfBarrelLayers;

    int i=0;
    thePixelPseudoRadDamage[i++] = conf.getParameter<double>("thePixelPseudoRadDamage_BPix1");
    thePixelPseudoRadDamage[i++] = conf.getParameter<double>("thePixelPseudoRadDamage_BPix2");
    thePixelPseudoRadDamage[i++] = conf.getParameter<double>("thePixelPseudoRadDamage_BPix3");
    thePixelPseudoRadDamage[i++] = conf.getParameter<double>("thePixelPseudoRadDamage_BPix4");
    
    // to be removed when Gaelle will have the phase2 digitizer
    if (NumberOfBarrelLayers>=5){
      if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
      // For Phase2 tracker layers just set the outermost BPix aging 0.
      for (int j=5 ; j<=NumberOfBarrelLayers ; j++){
	thePixelPseudoRadDamage[j-1]=0.;
      }
    }
    //
    i=FPixIndex;
    thePixelPseudoRadDamage[i++]   = conf.getParameter<double>("thePixelPseudoRadDamage_FPix1");
    thePixelPseudoRadDamage[i++]   = conf.getParameter<double>("thePixelPseudoRadDamage_FPix2");
    thePixelPseudoRadDamage[i++]   = conf.getParameter<double>("thePixelPseudoRadDamage_FPix3");
    
    //To be removed when Phase2 digitizer will be available
    if (NumberOfEndcapDisks>=4){
      if (NumberOfTotLayers>20){throw cms::Exception("Configuration") <<"SiPixelDigitizer was given more layers than it can handle";}
      // For Phase2 tracker layers just set the extra FPix disk aging to 0. BE CAREFUL THESE VALUES ARE HARDCODED ALSO ELSEWHERE IN THIS FILE
      for (int j=4+FPixIndex ; j<=NumberOfEndcapDisks+NumberOfBarrelLayers ; j++){
	thePixelPseudoRadDamage[j-1]=0.;
      }
    }
  }
  // the first "NumberOfBarrelLayers" settings [0],[1], ... , [NumberOfBarrelLayers-1] are for the barrel pixels
  // the next  "NumberOfEndcapDisks"  settings [NumberOfBarrelLayers],[NumberOfBarrelLayers+1], ... [NumberOfEndcapDisks+NumberOfBarrelLayers-1]
}

//=========================================================================
void SiPixelDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                                  std::vector<PSimHit>::const_iterator inputEnd,
						  const size_t inputBeginGlobalIndex,
						  const unsigned int tofBin,
                                                  const PixelGeomDetUnit* pixdet,
                                                  const GlobalVector& bfield,
						  const TrackerTopology *tTopo,
                                                  CLHEP::HepRandomEngine* engine) {
    // produce SignalPoint's for all SimHit's in detector
    // Loop over hits

    uint32_t detId = pixdet->geographicalId().rawId();
    size_t simHitGlobalIndex=inputBeginGlobalIndex; // This needs to stored to create the digi-sim link later
    for (std::vector<PSimHit>::const_iterator ssbegin = inputBegin; ssbegin != inputEnd; ++ssbegin, ++simHitGlobalIndex) {
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
      if (  ((*ssbegin).tof() - pixdet->surface().toGlobal((*ssbegin).localPosition()).mag()/30.)>= theTofLowerCut &&
	    ((*ssbegin).tof()- pixdet->surface().toGlobal((*ssbegin).localPosition()).mag()/30.) <= theTofUpperCut ) {
	primary_ionization(*ssbegin, ionization_points, engine); // fills _ionization_points
	drift(*ssbegin, pixdet, bfield, tTopo, ionization_points, collection_points);  // transforms _ionization_points to collection_points
	// compute induced signal on readout elements and add to _signal
	induce_signal(inputBegin, inputEnd, *ssbegin, simHitGlobalIndex, tofBin, pixdet, collection_points); // 1st 3 args needed only for SimHit<-->Digi link
      } //  end if
    } // end for

}

//============================================================================
void SiPixelDigitizerAlgorithm::calculateInstlumiFactor(PileupMixingContent* puInfo){
  //Instlumi scalefactor calculating for dynamic inefficiency

  if (puInfo) {
    const std::vector<int>& bunchCrossing = puInfo->getMix_bunchCrossing();
    const std::vector<float>& TrueInteractionList = puInfo->getMix_TrueInteractions();      
    //const int bunchSpacing = puInfo->getMix_bunchSpacing();

    int pui = 0, p = 0;
    std::vector<int>::const_iterator pu;
    std::vector<int>::const_iterator pu0 = bunchCrossing.end();
    
    for (pu=bunchCrossing.begin(); pu!=bunchCrossing.end(); ++pu) {
      if (*pu==0) {
	pu0 = pu;
	p = pui;
      }
      pui++;
    }
    if (pu0!=bunchCrossing.end()) {
      for (size_t i=0, n = pixelEfficiencies_.thePUEfficiency.size(); i<n; i++) {
	double instlumi = TrueInteractionList.at(p)*pixelEfficiencies_.theInstLumiScaleFactor;
	double instlumi_pow=1.;
	pixelEfficiencies_.pu_scale[i] = 0;
	for  (size_t j=0; j<pixelEfficiencies_.thePUEfficiency[i].size(); j++){
	  pixelEfficiencies_.pu_scale[i]+=instlumi_pow*pixelEfficiencies_.thePUEfficiency[i][j];
	  instlumi_pow*=instlumi;
	}
      }
    }
  } 
  else {
    for (int i=0, n = pixelEfficiencies_.thePUEfficiency.size(); i<n; i++) {
      pixelEfficiencies_.pu_scale[i] = 1.;
    }
  }
}

//============================================================================
void SiPixelDigitizerAlgorithm::calculateInstlumiFactor(const std::vector<PileupSummaryInfo> &ps, int bunchSpacing) {
  int p = -1;
  for(unsigned int i=0; i<ps.size(); i++)
    if (ps[i].getBunchCrossing() == 0)
      p=i;

  if(p >= 0) {
    for (size_t i=0, n = pixelEfficiencies_.thePUEfficiency.size(); i<n; i++) {
      double instlumi = ps[p].getTrueNumInteractions()*pixelEfficiencies_.theInstLumiScaleFactor;
      double instlumi_pow=1.;
      pixelEfficiencies_.pu_scale[i] = 0;
      for  (size_t j=0; j<pixelEfficiencies_.thePUEfficiency[i].size(); j++){
	pixelEfficiencies_.pu_scale[i]+=instlumi_pow*pixelEfficiencies_.thePUEfficiency[i][j];
	instlumi_pow*=instlumi;
      }
    }
  }
  else {
    for (int i=0, n = pixelEfficiencies_.thePUEfficiency.size(); i<n; i++) {
      pixelEfficiencies_.pu_scale[i] = 1.;
    }
  }
}

//============================================================================
void SiPixelDigitizerAlgorithm::setSimAccumulator(const std::map<uint32_t, std::map<int, int> >& signalMap) {
  for(const auto& det: signalMap) {
    auto& theSignal = _signal[det.first];
    for(const auto& chan: det.second) {
      theSignal[chan.first].set(chan.second * theElectronPerADC); // will get divided again by theElectronPerAdc in digitize...
    }
  }
}

//============================================================================
void SiPixelDigitizerAlgorithm::digitize(const PixelGeomDetUnit* pixdet,
                                         std::vector<PixelDigi>& digis,
                                         std::vector<PixelDigiSimLink>& simlinks,
					 const TrackerTopology *tTopo,
                                         CLHEP::HepRandomEngine* engine) {
  
  // Pixel Efficiency moved from the constructor to this method because
  // the information of the det are not available in the constructor
  // Efficiency parameters. 0 - no inefficiency, 1-low lumi, 10-high lumi
  
  uint32_t detID = pixdet->geographicalId().rawId();
  const signal_map_type& theSignal = _signal[detID];
  
  // Noise already defined in electrons
  // thePixelThresholdInE = thePixelThreshold * theNoiseInElectrons ;
  // Find the threshold in noise units, needed for the noiser.
  

  float thePixelThresholdInE = 0.;
  
  if(theNoiseInElectrons>0.){
    if(pixdet->type().isTrackerPixel() && pixdet->type().isBarrel()){ // Barrel modules
      int lay = tTopo->layer(detID);
      if(addThresholdSmearing) {
	if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel || pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXB) {
	  if (lay==1) {
	    thePixelThresholdInE = CLHEP::RandGaussQ::shoot(engine, theThresholdInE_BPix_L1, theThresholdSmearing_BPix_L1); // gaussian smearing
	  } else if (lay==2) {
	    thePixelThresholdInE = CLHEP::RandGaussQ::shoot(engine, theThresholdInE_BPix_L2, theThresholdSmearing_BPix_L2); // gaussian smearing
	  } else {
	    thePixelThresholdInE = CLHEP::RandGaussQ::shoot(engine, theThresholdInE_BPix , theThresholdSmearing_BPix); // gaussian smearing
	  }
	}
      } else {
	if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel || pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXB) {
	  if (lay==1) {
	    thePixelThresholdInE = theThresholdInE_BPix_L1;
	  } else if (lay==2) {
	    thePixelThresholdInE = theThresholdInE_BPix_L2;
	  } else {
	    thePixelThresholdInE = theThresholdInE_BPix; // no smearing
	  }
	}
      }
    } else if(pixdet->type().isTrackerPixel()) { // Forward disks modules
      if(addThresholdSmearing) {
	thePixelThresholdInE = CLHEP::RandGaussQ::shoot(engine, theThresholdInE_FPix, theThresholdSmearing_FPix); // gaussian smearing
      } else {
	thePixelThresholdInE = theThresholdInE_FPix; // no smearing
      }
    }
    else {throw cms::Exception("NotAPixelGeomDetUnit") << "Not a pixel geomdet unit" << detID;}
  }
  

#ifdef TP_DEBUG
  const PixelTopology* topol=&pixdet->specificTopology();  
  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();
  // full detector thickness
  float moduleThickness = pixdet->specificSurface().bounds().thickness();
  LogDebug ("PixelDigitizer")
    << " PixelDigitizer "
    << numColumns << " " << numRows << " " << moduleThickness;
#endif
  
  if(addNoise) add_noise(pixdet, thePixelThresholdInE/theNoiseInElectrons, engine);  // generate noise
  
  // Do only if needed
  
  if((AddPixelInefficiency) && (!theSignal.empty()))
    pixel_inefficiency(pixelEfficiencies_, pixdet, tTopo, engine); // Kill some pixels
  
  if(use_ineff_from_db_ && (!theSignal.empty()))
    pixel_inefficiency_db(detID);
  
  if(use_module_killing_) {
    if (use_deadmodule_DB_) {  // remove dead modules using DB
      module_killing_DB(detID);
    } else { // remove dead modules using the list in cfg file
      module_killing_conf(detID);
    }
  }
  
  make_digis(thePixelThresholdInE, detID, pixdet, digis, simlinks, tTopo);
  
#ifdef TP_DEBUG
  LogDebug ("PixelDigitizer") << "[SiPixelDigitizerAlgorithm] converted " << digis.size() << " PixelDigis in DetUnit" << detID;
#endif
}

//***********************************************************************/
// Generate primary ionization along the track segment.
// Divide the track into small sub-segments
void SiPixelDigitizerAlgorithm::primary_ionization(const PSimHit& hit, std::vector<EnergyDepositUnit>& ionization_points, CLHEP::HepRandomEngine* engine) const {
  
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
		   elossVector, engine);
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
                                               int NumberOfSegs,float elossVector[],
                                               CLHEP::HepRandomEngine* engine) const {
  
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
                                       segmentEloss, engine )/1000.; //convert to GeV 
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
				      const TrackerTopology *tTopo,
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
    
    // Insert a charge loss due to Rad Damage here
    float energyOnCollector = ionization_points[i].energy(); // The energy that reaches the collector
    
    // add pixel aging 
    if (AddPixelAging) {
      float kValue = pixel_aging(pixelAging_,pixdet,tTopo);
      energyOnCollector *= exp( -1*kValue*DriftDistance/moduleThickness );
    }
    
#ifdef TP_DEBUG
    LogDebug ("Pixel Digitizer")
      <<" Dift DistanceZ= "<<DriftDistance<<" module thickness= "<<moduleThickness
      <<" Start Energy= "<<ionization_points[i].energy()<<" Energy after loss= "<<energyOnCollector;
#endif
    SignalPoint sp( CloudCenterX, CloudCenterY,
		    Sigma_x, Sigma_y, hit.tof(), energyOnCollector );
    
    // Load the Charge distribution parameters
    collection_points[i] = (sp);
    
  } // loop over ionization points, i.
  
} // end drift

//*************************************************************************
// Induce the signal on the collection plane of the active sensor area.
void SiPixelDigitizerAlgorithm::induce_signal(std::vector<PSimHit>::const_iterator inputBegin,
					      std::vector<PSimHit>::const_iterator inputEnd,
					      const PSimHit& hit,
					      const size_t hitIndex,
					      const unsigned int tofBin,
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
     // PIXEL AREA IS LOST, it should not be collected.

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

     // First integrate charge strips in x
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
	 LowerBound = 1-calcQ((xLB-CloudCenterX)/SigmaX);
       }

       if(ix == numRows-1 || SigmaX==0. )
	 UpperBound = 1.;
       else {
	 mp = MeasurementPoint( float(ix+1), 0.0);
	 xUB = topol->localPosition(mp).x();
	 UpperBound = 1. - calcQ((xUB-CloudCenterX)/SigmaX);
       }

       float   TotalIntegrationRange = UpperBound - LowerBound; // get strip
       x[ix] = TotalIntegrationRange; // save strip integral
       //if(SigmaX==0 || SigmaY==0)
       //cout<<TotalIntegrationRange<<" "<<ix<<std::endl;

     }

    // Now integrate strips in y
    int iy; // TT for compatibility
    for (iy=IPixLeftDownY; iy<=IPixRightUpY; iy++) { //loope over y ind
      float yUB, yLB, UpperBound, LowerBound;

      if(iy == 0 || SigmaY==0.)
	LowerBound = 0.;
      else {
        mp = MeasurementPoint( 0.0, float(iy) );
        yLB = topol->localPosition(mp).y();
	LowerBound = 1. - calcQ((yLB-CloudCenterY)/SigmaY);
      }

      if(iy == numColumns-1 || SigmaY==0. )
	UpperBound = 1.;
      else {
        mp = MeasurementPoint( 0.0, float(iy+1) );
        yUB = topol->localPosition(mp).y();
	UpperBound = 1. - calcQ((yUB-CloudCenterY)/SigmaY);
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

#ifdef TP_DEBUG
	mp = MeasurementPoint( float(ix), float(iy) );
	LocalPoint lp = topol->localPosition(mp);
	chan = topol->channel(lp);
	LogDebug ("Pixel Digitizer")
	  << " pixel " << ix << " " << iy << " - "<<" "
	  << chan << " " << ChargeFraction<<" "
	  << mp.x() << " " << mp.y() <<" "
	  << lp.x() << " " << lp.y() << " "  // givex edge position
	  << chan; // edge belongs to previous ?
#endif

      } // endfor iy
    } //endfor ix


  } // loop over charge distributions

  // Fill the global map with all hit pixels from this event

   bool reweighted = false;
   if (UseReweighting){
     if(hit.processType()==0){
       reweighted = hitSignalReweight (hit, hit_signal, hitIndex, tofBin, topol, detID, theSignal, hit.processType());
     }else{
       // If it's not the primary particle, use the first hit in the collection as SimHit, which should be the corresponding primary.
       reweighted = hitSignalReweight ((*inputBegin), hit_signal, hitIndex, tofBin, topol, detID, theSignal, hit.processType());
     }
   }
   if (!reweighted){
     for ( hit_map_type::const_iterator im = hit_signal.begin();
	   im != hit_signal.end(); ++im) {
       int chan =  (*im).first;
       theSignal[chan] += (makeDigiSimLinks_ ? Amplitude( (*im).second, &hit, hitIndex, tofBin, (*im).second) : Amplitude( (*im).second, (*im).second) )  ;
       
#ifdef TP_DEBUG
       std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
       LogDebug ("Pixel Digitizer")
	 << " pixel " << ip.first << " " << ip.second << " "
	 << theSignal[chan];
#endif
     }
   }

} // end induce_signal

/***********************************************************************/

// Build pixels, check threshold, add misscalibration, ...
void SiPixelDigitizerAlgorithm::make_digis(float thePixelThresholdInE,
                                           uint32_t detID,
					   const PixelGeomDetUnit* pixdet,
                                           std::vector<PixelDigi>& digis,
                                           std::vector<PixelDigiSimLink>& simlinks,
					   const TrackerTopology *tTopo) const  {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " make digis "<<" "
			       << " pixel threshold FPix" << theThresholdInE_FPix << " "
                               << " pixel threshold BPix" << theThresholdInE_BPix << " "
                               << " pixel threshold BPix Layer1" << theThresholdInE_BPix_L1 << " "
                               << " pixel threshold BPix Layer2" << theThresholdInE_BPix_L2 << " "
			       << " List pixels passing threshold ";
#endif

  // Loop over hit pixels

  signalMaps::const_iterator it = _signal.find(detID);
  if (it == _signal.end()) {
    return;
  }

  const signal_map_type& theSignal = (*it).second;

  // unsigned long is enough to store SimTrack id and EncodedEventId
  using TrackEventId = std::pair<decltype(SimTrack().trackId()), decltype(EncodedEventId().rawId())>;
  std::map<TrackEventId, float> simi; // re-used

  for (signal_map_const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {

    float signalInElectrons = (*i).second ;   // signal in electrons

    // Do the miss calibration for calibration studies only.
    //if(doMissCalibrate) signalInElectrons = missCalibrate(signalInElectrons)

    // Do only for pixels above threshold

    if( signalInElectrons >= thePixelThresholdInE && signalInElectrons > 0.) { // check threshold, always reject killed (0-charge) digis

      int chan =  (*i).first;  // channel number
      std::pair<int,int> ip = PixelDigi::channelToPixel(chan);
      int adc=0;  // ADC count as integer

      // Do the miss calibration for calibration studies only.
      if(doMissCalibrate) {
	int row = ip.first;  // X in row
	int col = ip.second; // Y is in col
	adc = int(missCalibrate(detID, tTopo, pixdet, col, row, signalInElectrons)); //full misscalib.
      } else { // Just do a simple electron->adc conversion
	adc = int( signalInElectrons / theElectronPerADC ); // calibrate gain
      }
      adc = std::min(adc, theAdcFullScale); // Check maximum value
#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer")
	<< (*i).first << " " << (*i).second << " " << signalInElectrons
	<< " " << adc << ip.first << " " << ip.second ;
#endif

      // Load digis
      digis.emplace_back(ip.first, ip.second, adc);

      if (makeDigiSimLinks_ && !(*i).second.hitInfos().empty()) {
        //digilink
	  unsigned int il=0;
          for(const auto& info: (*i).second.hitInfos()) {
            // note: according to C++ standard operator[] does
            // value-initializiation, which for float means initial value of 0
            simi[std::make_pair(info.trackId(), info.eventId().rawId())] += (*i).second.individualampl()[il];
            il++;
          }

	  //sum the contribution of the same trackid
          for(const auto& info: (*i).second.hitInfos()) {
            // skip if track already processed
            auto found = simi.find(std::make_pair(info.trackId(), info.eventId().rawId()));
            if(found == simi.end())
              continue;

	    float sum_samechannel = found->second;
	    float fraction=sum_samechannel/(*i).second;
	    if(fraction>1.f) fraction=1.f;

            // Approximation: pick hitIndex and tofBin only from the first SimHit
	    simlinks.emplace_back((*i).first, info.trackId(), info.hitIndex(), info.tofBin(), info.eventId(), fraction);
            simi.erase(found);
	  }
          simi.clear(); // although should be empty already
      }
    }
  }
}

/***********************************************************************/

//  Add electronic noise to pixel charge
void SiPixelDigitizerAlgorithm::add_noise(const PixelGeomDetUnit* pixdet,
                                          float thePixelThreshold,
                                          CLHEP::HepRandomEngine* engine) {

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
        float noise_ChargeVCALSmearing = theSmearedChargeRMS * CLHEP::RandGaussQ::shoot(engine, 0., 1.);
	// Noise from full readout:
        float noise  = CLHEP::RandGaussQ::shoot(engine, 0., theReadoutNoise);

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
	float noise = CLHEP::RandGaussQ::shoot(engine, 0., theReadoutNoise);

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
                      otherPixels,
                      engine );

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
						   const TrackerTopology *tTopo,
                                                   CLHEP::HepRandomEngine* engine) {
  
  uint32_t detID= pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  const PixelTopology* topol=&pixdet->specificTopology();
  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();
  bool isPhase1 = pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXB 
    || pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXEC;
  // Predefined efficiencies
  double pixelEfficiency  = 1.0;
  double columnEfficiency = 1.0;
  double chipEfficiency   = 1.0;
  std::vector<double> pixelEfficiencyROCStdPixels(16,1);
  std::vector<double> pixelEfficiencyROCBigPixels(16,1);
  
  if (eff.FromConfig) {
    // setup the chip indices conversion
    if    (pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel ||
           pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXB){// barrel layers
      int layerIndex=tTopo->layer(detID);
      pixelEfficiency  = eff.thePixelEfficiency[layerIndex-1];
      columnEfficiency = eff.thePixelColEfficiency[layerIndex-1];
      chipEfficiency   = eff.thePixelChipEfficiency[layerIndex-1];
      //std::cout <<"Using BPix columnEfficiency = "<<columnEfficiency<< " for layer = "<<layerIndex <<"\n";
      // This should never happen, but only check if it is not an upgrade geometry
      if (NumberOfBarrelLayers==3){
        if(numColumns>416)  LogWarning ("Pixel Geometry") <<" wrong columns in barrel "<<numColumns;
        if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in barrel "<<numRows;
        
        int ladder=tTopo->pxbLadder(detID);
        int module=tTopo->pxbModule(detID);
        if (module<=4) module=5-module;
        else module-=4;
        
        columnEfficiency *= eff.theLadderEfficiency_BPix[layerIndex-1][ladder-1]*eff.theModuleEfficiency_BPix[layerIndex-1][module-1]*eff.pu_scale[layerIndex-1];
      }
    } else if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::PixelEndcap ||
              pixdet->subDetector()==GeomDetEnumerators::SubDetector::P1PXEC ||
              pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2PXEC){                // forward disks
    
      unsigned int diskIndex=tTopo->layer(detID)+eff.FPixIndex; // Use diskIndex-1 later to stay consistent with BPix
      unsigned int panelIndex=tTopo->pxfPanel(detID);
      unsigned int moduleIndex=tTopo->pxfModule(detID);
      //if (eff.FPixIndex>diskIndex-1){throw cms::Exception("Configuration") <<"SiPixelDigitizer is using the wrong efficiency value. index = "
      //                                                                       <<diskIndex-1<<" , MinIndex = "<<eff.FPixIndex<<" ... "<<tTopo->pxfDisk(detID);}
      pixelEfficiency  = eff.thePixelEfficiency[diskIndex-1];
      columnEfficiency = eff.thePixelColEfficiency[diskIndex-1];
      chipEfficiency   = eff.thePixelChipEfficiency[diskIndex-1];
      //std::cout <<"Using FPix columnEfficiency = "<<columnEfficiency<<" for Disk = "<< tTopo->pxfDisk(detID)<<"\n";
      // Sometimes the forward pixels have wrong size,
      // this crashes the index conversion, so exit, but only check if it is not an upgrade geometry
      if (NumberOfBarrelLayers==3){  // whether it is the present or the phase 1 detector can be checked using GeomDetEnumerators::SubDetector
        if(numColumns>260 || numRows>160) {
          if(numColumns>260)  LogWarning ("Pixel Geometry") <<" wrong columns in endcaps "<<numColumns;
          if(numRows>160)  LogWarning ("Pixel Geometry") <<" wrong rows in endcaps "<<numRows;
          return;
        }
        if ((panelIndex==1 && (moduleIndex==1 || moduleIndex==2)) || (panelIndex==2 && moduleIndex==1)) { //inner modules
          columnEfficiency*=eff.theInnerEfficiency_FPix[diskIndex-1]*eff.pu_scale[3];
        } else { //outer modules
          columnEfficiency*=eff.theOuterEfficiency_FPix[diskIndex-1]*eff.pu_scale[4];
        }
      } // current detector, forward
    } else if(pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2OTB ||pixdet->subDetector()==GeomDetEnumerators::SubDetector::P2OTEC) {
      // If phase 2 outer tracker, hardcoded values as they have been so far
      pixelEfficiency  = 0.999;
      columnEfficiency = 0.999;
      chipEfficiency   = 0.999;
    } // if barrel/forward
  } else { // Load precomputed factors from Database
    pixelEfficiency  = eff.PixelGeomFactors.at(detID);
    columnEfficiency = eff.ColGeomFactors.at(detID)*eff.pu_scale[eff.iPU.at(detID)];
    chipEfficiency   = eff.ChipGeomFactors.at(detID);
    if (isPhase1){
      for (unsigned int i_roc=0; i_roc<eff.PixelGeomFactorsROCStdPixels.at(detID).size();++i_roc){
	pixelEfficiencyROCStdPixels[i_roc] = eff.PixelGeomFactorsROCStdPixels.at(detID).at(i_roc);
	pixelEfficiencyROCBigPixels[i_roc] = eff.PixelGeomFactorsROCBigPixels.at(detID).at(i_roc);    
      }
    } // is Phase 1
  }
  
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter pixel_inefficiency " << pixelEfficiency << " "
			       << columnEfficiency << " " << chipEfficiency;
#endif
  
  // Initilize the index converter
  //PixelIndices indexConverter(numColumns,numRows);
  std::unique_ptr<PixelIndices> pIndexConverter(new PixelIndices(numColumns,numRows));

  int chipIndex = 0;
  int rowROC = 0;
  int colROC = 0;
  std::map<int, int, std::less<int> >chips, columns, pixelStd, pixelBig;
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
    if (isPhase1){
      if (topol->isItBigPixelInX(row) || topol->isItBigPixelInY(col))    
	pixelBig[chipIndex]++;
      else
	pixelStd[chipIndex]++;
    }
  }
  
  // Delete some ROC hits.
  for ( iter = chips.begin(); iter != chips.end() ; iter++ ) {
    //float rand  = RandFlat::shoot();
    float rand  = CLHEP::RandFlat::shoot(engine);
    if( rand > chipEfficiency ) chips[iter->first]=0;
  }
  
  // Delete some Dcol hits.
  for ( iter = columns.begin(); iter != columns.end() ; iter++ ) {
    //float rand  = RandFlat::shoot();
    float rand  = CLHEP::RandFlat::shoot(engine);
    if( rand > columnEfficiency ) columns[iter->first]=0;
  }
  
  // Delete some pixel hits based on DCDC issue damage.
  if (isPhase1){
    for ( iter = pixelStd.begin(); iter != pixelStd.end() ; iter++ ) {
      float rand  = CLHEP::RandFlat::shoot(engine);    
      if( rand > pixelEfficiencyROCStdPixels[iter->first]) pixelStd[iter->first] = 0;
    }
    
    for ( iter = pixelBig.begin(); iter != pixelBig.end() ; iter++ ) {
      float rand  = CLHEP::RandFlat::shoot(engine);    
      if( rand > pixelEfficiencyROCBigPixels[iter->first]) pixelBig[iter->first] = 0;
    }
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
    float rand  = CLHEP::RandFlat::shoot(engine);
    if( chips[chipIndex]==0 || columns[dColInDet]==0
	|| rand>pixelEfficiency  
	|| (pixelStd.count(chipIndex) && pixelStd[chipIndex] == 0)
	|| (pixelBig.count(chipIndex) && pixelBig[chipIndex] == 0)) {
      // make pixel amplitude =0, pixel will be lost at clusterization
      i->second.set(0.); // reset amplitude,      
    } // end if
    if (isPhase1){
      if((pixelStd.count(chipIndex) && pixelStd[chipIndex] == 0)
	 || (pixelBig.count(chipIndex) && pixelBig[chipIndex] == 0)) {
	// make pixel amplitude =0, pixel will be lost at clusterization
	i->second.set(0.); // reset amplitude,      
      } // end if
    } // is Phase 1
  } // end pixel loop
} // end pixel_indefficiency

//***************************************************************************************
// Simulate pixel aging with an exponential function
//**************************************************************************************

float SiPixelDigitizerAlgorithm::pixel_aging(const PixelAging& aging,
					     const PixelGeomDetUnit *pixdet,
					     const TrackerTopology *tTopo) const {
  
  uint32_t detID= pixdet->geographicalId().rawId();
  
  
  // Predefined damage parameter (no aging)
  float pseudoRadDamage  = 0.0f;
  
  // setup the chip indices conversion
  if    (pixdet->subDetector() ==  GeomDetEnumerators::SubDetector::PixelBarrel ||
	 pixdet->subDetector() ==  GeomDetEnumerators::SubDetector::P1PXB){// barrel layers
    int layerIndex=tTopo->layer(detID);
    
     pseudoRadDamage  = aging.thePixelPseudoRadDamage[layerIndex-1];
     
     //  std::cout << "pixel_aging: " << std::endl;
     // std::cout << "Subid " << Subid << " layerIndex " << layerIndex << " ladder " << tTopo->pxbLadder(detID)  << " module  " << tTopo->pxbModule(detID) << std::endl;
     
  } else if (pixdet->subDetector() == GeomDetEnumerators::SubDetector::PixelEndcap ||
	     pixdet->subDetector() == GeomDetEnumerators::SubDetector::P1PXEC ||
	     pixdet->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC) {                // forward disks
    unsigned int diskIndex=tTopo->layer(detID)+aging.FPixIndex; // Use diskIndex-1 later to stay consistent with BPix
    
    pseudoRadDamage  = aging.thePixelPseudoRadDamage[diskIndex-1];
    
    //    std::cout << "pixel_aging: " << std::endl;
    //    std::cout << "Subid " << Subid << " diskIndex " << diskIndex << std::endl;
  } else if (pixdet->subDetector() == GeomDetEnumerators::SubDetector::P2OTB || pixdet->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC) {
    // if phase 2 OT hardcoded value as it has always been
    pseudoRadDamage = 0.f;
  } // if barrel/forward
  
  //  std::cout << " pseudoRadDamage " << pseudoRadDamage << std::endl;
  //  std::cout << " end pixel_aging " << std::endl;
  
  return pseudoRadDamage;
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter pixel_aging " << pseudoRadDamage;
#endif
  
}

//***********************************************************************

// Fluctuate the gain and offset for the amplitude calibration
// Use gaussian smearing.
//float SiPixelDigitizerAlgorithm::missCalibrate(const float amp) const {
  //float gain  = RandGaussQ::shoot(1.,theGainSmearing);
  //float offset  = RandGaussQ::shoot(0.,theOffsetSmearing);
  //float newAmp = amp * gain + offset;
  // More complex misscalibration
float SiPixelDigitizerAlgorithm::missCalibrate(uint32_t detID, const TrackerTopology *tTopo, const PixelGeomDetUnit* pixdet, int col,int row,
				 const float signalInElectrons) const {
  // Central values
  //const float p0=0.00352, p1=0.868, p2=112., p3=113.; // pix(0,0,0)
  //  const float p0=0.00382, p1=0.886, p2=112.7, p3=113.0; // average roc=0
  //const float p0=0.00492, p1=1.998, p2=90.6, p3=134.1; // average roc=6
  // Smeared (rms)
  //const float s0=0.00020, s1=0.051, s2=5.4, s3=4.4; // average roc=0
  //const float s0=0.00015, s1=0.043, s2=3.2, s3=3.1; // col average roc=0

  // Make 2 sets of parameters for Fpix and BPIx:

  float p0=0.0f;
  float p1=0.0f;
  float p2=0.0f;
  float p3=0.0f;

  if(pixdet->type().isTrackerPixel() && pixdet->type().isBarrel()){// barrel layers
      p0 = BPix_p0;
      p1 = BPix_p1;
      p2 = BPix_p2;
      p3 = BPix_p3;
  } else if(pixdet->type().isTrackerPixel()) {// forward disks
      p0 = FPix_p0;
      p1 = FPix_p1;
      p2 = FPix_p2;
      p3 = FPix_p3;
  } else {
    throw cms::Exception("NotAPixelGeomDetUnit") << "Not a pixel geomdet unit" << detID;
  }

  float newAmp = 0.f; //Modified signal

  // Convert electrons to VCAL units
  float signal = (signalInElectrons-electronsPerVCAL_Offset)/electronsPerVCAL;

  // New gains/offsets are needed for phase1 L1
  int layer = 0;
  if (DetId(detID).subdetId()==1) layer = tTopo->pxbLayer(detID);
  if (layer==1) signal = (signalInElectrons-electronsPerVCAL_L1_Offset)/electronsPerVCAL_L1;

  // Simulate the analog response with fixed parametrization
  newAmp = p3 + p2 * tanh(p0*signal - p1);


  // Use the pixel-by-pixel calibrations
  //transform to ROC index coordinates
  //int chipIndex=0, colROC=0, rowROC=0;
  //std::unique_ptr<PixelIndices> pIndexConverter(new PixelIndices(numColumns,numRows));
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

  float dir_x = 0.0f;
  float dir_y = 0.0f;
  float dir_z = 0.0f;
  float scale = 0.0f;

  uint32_t detID= pixdet->geographicalId().rawId();


  // Read Lorentz angle from cfg file:**************************************************************

  if(!use_LorentzAngle_DB_){
    
    if( alpha2Order) {
      alpha2_FPix = tanLorentzAnglePerTesla_FPix*tanLorentzAnglePerTesla_FPix;
      alpha2_BPix = tanLorentzAnglePerTesla_BPix*tanLorentzAnglePerTesla_BPix;
    }else {
      alpha2_FPix = 0.0f;
      alpha2_BPix = 0.0f;
    }
    
    if(pixdet->type().isTrackerPixel() && pixdet->type().isBarrel()){// barrel layers
      dir_x = -( tanLorentzAnglePerTesla_BPix * Bfield.y() + alpha2_BPix* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_BPix * Bfield.x() - alpha2_BPix* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_BPix* Bfield.z()*Bfield.z() );
      scale = -dir_z;
    } else if (pixdet->type().isTrackerPixel()) {// forward disks
      dir_x = -( tanLorentzAnglePerTesla_FPix * Bfield.y() + alpha2_FPix* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_FPix * Bfield.x() - alpha2_FPix* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_FPix* Bfield.z()*Bfield.z() );
      scale = -dir_z;
    } else {
      throw cms::Exception("NotAPixelGeomDetUnit") << "Not a pixel geomdet unit" << detID;
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
    scale = -dir_z;
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
	  if((std::abs(ip.second - it->col) < 26) ) {i->second.set(0.);}
          else if(it->row==120 && ip.second-it->col==26){i->second.set(0.);}
          else if(it->row==119 && it->col-ip.second==26){i->second.set(0.);}
	}
	else if(it->row < 80 && ip.first < 80 ){
	  if((std::abs(ip.second - it->col) < 26) ){i->second.set(0.);}
          else if(it->row==40 && ip.second-it->col==26){i->second.set(0.);}
          else if(it->row==39 && it->col-ip.second==26){i->second.set(0.);}
       }
      }
    }
  }
}



bool SiPixelDigitizerAlgorithm::hitSignalReweight(const PSimHit& hit, 
						  std::map< int, float, std::less<int> >& hit_signal, 
						  const size_t hitIndex,
						  const unsigned int tofBin,
						  const PixelTopology* topol,
						  uint32_t detID,
						  signal_map_type& theSignal,
						  unsigned short int processType){

  int irow_min = topol->nrows();
  int irow_max = 0;
  int icol_min = topol->ncolumns();
  int icol_max = 0;
     
  float chargeBefore = 0;
  float chargeAfter = 0;
  signal_map_type hitSignal;
  LocalVector direction = hit.exitPoint() - hit.entryPoint();
  
  for ( std::map< int, float, std::less<int> >::const_iterator im = hit_signal.begin(); im != hit_signal.end(); ++im) {
    int chan =  (*im).first;
    std::pair<int,int> pixelWithCharge = PixelDigi::channelToPixel( chan);
    //std::cout << "PixelHit - x: " << pixelWithCharge.first << " y: " << pixelWithCharge.second << "  With Charge:  " << (*im).second <<  std::endl;
    
    hitSignal[chan] += (makeDigiSimLinks_ ? Amplitude( (*im).second, &hit, hitIndex, tofBin, (*im).second) : Amplitude( (*im).second, (*im).second) )  ;
    chargeBefore += (*im).second;
    
    if(pixelWithCharge.first < irow_min)
      irow_min = pixelWithCharge.first;
    if(pixelWithCharge.first > irow_max)
      irow_max = pixelWithCharge.first;
    if(pixelWithCharge.second < icol_min)
      icol_min = pixelWithCharge.second;
    if(pixelWithCharge.second > icol_max)
      icol_max = pixelWithCharge.second;
  }
  
  LocalPoint hitEntryPoint = hit.entryPoint();
  
  float trajectoryScaleToPosition = hitEntryPoint.z()/direction.z();
  
  if( (hitEntryPoint.z() > 0 && direction.z() < 0) || (hitEntryPoint.z() < 0 && direction.z() > 0) ){
    trajectoryScaleToPosition *= -1;
  }
  
  LocalPoint hitPosition = hitEntryPoint + trajectoryScaleToPosition * direction;  

  MeasurementPoint hitPositionPixel = topol->measurementPosition(hit.localPosition() );
  std::pair<int,int> hitPixel = std::pair<int,int>( int( floor(hitPositionPixel.x() ) ), int ( floor(hitPositionPixel.y() ) ));
  
  MeasurementPoint originPixel = MeasurementPoint(hitPixel.first - THX + 0.5, hitPixel.second - THY + 0.5);
  LocalPoint origin = topol->localPosition(originPixel);
  
  MeasurementPoint hitEntryPointPixel = topol->measurementPosition(hit.entryPoint() );
  MeasurementPoint hitExitPointPixel = topol->measurementPosition(hit.exitPoint() );
  std::pair<int,int> entryPixel = std::pair<int,int>( int( floor(hitEntryPointPixel.x() ) ), int ( floor(hitEntryPointPixel.y() ) ));
  std::pair<int,int> exitPixel = std::pair<int,int>( int( floor(hitExitPointPixel.x() ) ), int ( floor(hitExitPointPixel.y() ) ));

  int hitcol_min, hitcol_max, hitrow_min, hitrow_max;
  if(entryPixel.first>exitPixel.first){
    hitrow_min = exitPixel.first;
    hitrow_max = entryPixel.first;
  }else{
    hitrow_min = entryPixel.first;
    hitrow_max = exitPixel.first;
  }

  if(entryPixel.second>exitPixel.second){
    hitcol_min = exitPixel.second;
    hitcol_max = entryPixel.second;
  }else{
    hitcol_min = entryPixel.second;
    hitcol_max = exitPixel.second;
  }

  
#ifdef TP_DEBUG
  LocalPoint CMSSWhitPosition = hit.localPosition();

  LogDebug ("Pixel Digitizer")
  << "\n"
  << "Particle ID is: " << hit.particleType() << "\n"
  << "Process type: " << hit.processType() << "\n"
  << "HitPosition:" << "\n"
  << "Hit entry x/y/z: " << hit.entryPoint().x() << "  " << hit.entryPoint().y() << "  " << hit.entryPoint().z() << "  "
  << "Hit exit x/y/z: " << hit.exitPoint().x() << "  " << hit.exitPoint().y() << "  " << hit.exitPoint().z() << "  "

  << "Pixel Pos - X: " << hitPositionPixel.x() << " Y: " << hitPositionPixel.y() << "\n"
  << "Cart.Cor. - X: " << CMSSWhitPosition.x() << " Y: " << CMSSWhitPosition.y() << "\n"
  << "Z=0 Pos - X: " << hitPosition.x() << " Y: " << hitPosition.y() << "\n"
  
  << "Origin of the template:" << "\n"
  << "Pixel Pos - X: " << originPixel.x() << " Y: " << originPixel.y() << "\n"
  << "Cart.Cor. - X: " << origin.x() << " Y: " << origin.y() << "\n"
  << "\n"
  << "Entry/Exit:" << "\n"
  << "Entry - X: " << hit.entryPoint().x() << " Y: " << hit.entryPoint().y() << " Z: " << hit.entryPoint().z() << "\n"
  << "Exit - X: " << hit.exitPoint().x() << " Y: " << hit.exitPoint().y() << " Z: " << hit.exitPoint().z() << "\n"
  
  << "Entry - X Pixel: " << hitEntryPointPixel.x() << " Y Pixel: " << hitEntryPointPixel.y() << "\n"
  << "Exit - X Pixel: " << hitExitPointPixel.x() << " Y Pixel: " << hitExitPointPixel.y() << "\n"
  
  << "row min: " << irow_min << " col min: " << icol_min << "\n";
#endif

  if(!(irow_min<=hitrow_max && irow_max>=hitrow_min && icol_min<=hitcol_max && icol_max>=hitcol_min)){
    // The clusters do not have an overlap, hence the hit is NOT reweighted
    return false;
  }

  
  float cmToMicrons = 10000.f;
  
  track[0] = (hitPosition.x() - origin.x() )*cmToMicrons;
  track[1] = (hitPosition.y() - origin.y() )*cmToMicrons;
  track[2] = 0.0f; //Middle of sensor is origin for Z-axis
  track[3] = direction.x();
  track[4] = direction.y();
  track[5] = direction.z();
  
  array_2d pixrewgt(boost::extents[TXSIZE][TYSIZE]);

  for(int row = 0; row < TXSIZE; ++row) {
    for(int col = 0; col < TYSIZE; ++col) {
      pixrewgt[row][col] = 0;
    }
  }
  
  for(int row = 0; row < TXSIZE; ++row) {
    xdouble[row] = topol->isItBigPixelInX(hitPixel.first + row - THX);
  }
  
  for(int col = 0; col < TYSIZE; ++col) {
    ydouble[col] = topol->isItBigPixelInY(hitPixel.second + col - THY);
  }

  for(int row = 0; row < TXSIZE; ++row) {
    for(int col = 0; col < TYSIZE; ++col) {
      //Fill charges into 21x13 Pixel Array with hitPixel in centre
      pixrewgt[row][col] = hitSignal[PixelDigi::pixelToChannel(hitPixel.first + row - THX, hitPixel.second + col - THY)];
      //std::cout << "Signal in " << hitPixel.first + row - THX << "/" << hitPixel.second + col - THY << " is " << hitSignal[PixelDigi::pixelToChannel(hitPixel.first + row - THX, hitPixel.second + col - THY)] << std::endl;
    }
  }
    
  if(PrintClusters){
    std::cout << "Cluster before reweighting: " << std::endl;
    printCluster(pixrewgt);  
  }

  int ierr;
  // for unirradiated: 2nd argument is IDden
  // for irradiated: 2nd argument is IDnum
  if (UseReweighting == true){
    int ID1 = dbobject_num->getTemplateID(detID);
    int ID0 = dbobject_den->getTemplateID(detID);

    if(ID0==ID1){
      return false;
    }
    ierr = PixelTempRewgt2D(ID0, ID1, pixrewgt);
  }
  else{
    ierr = PixelTempRewgt2D(IDden, IDden, pixrewgt);
  }
  if (ierr!=0){
#ifdef TP_DEBUG 
    LogDebug ("PixelDigitizer ") << "Cluster Charge Reweighting did not work properly.";
#endif
    return false;
  }
  
  if(PrintClusters){
    std::cout << "Cluster after reweighting: " << std::endl;  
    printCluster(pixrewgt);
  }
  
  for(int row = 0; row < TXSIZE; ++row) { 
    for(int col = 0; col < TYSIZE; ++col) {
      float charge = 0;
      charge = pixrewgt[row][col];
      if( (hitPixel.first + row - THX) >= 0 && (hitPixel.first + row - THX) < topol->nrows() && (hitPixel.second + col - THY) >= 0 && (hitPixel.second + col - THY) < topol->ncolumns() && charge > 0){
	chargeAfter += charge;
	theSignal[PixelDigi::pixelToChannel(hitPixel.first + row - THX, hitPixel.second + col - THY)] += (makeDigiSimLinks_ ? Amplitude(charge , &hit, hitIndex, tofBin, charge) : Amplitude( charge, charge) )  ;
      }
    }
  }

  if(chargeBefore!=0. && chargeAfter==0.){
    return false;
  }
  
  if(PrintClusters){
    std::cout << std::endl;
    std::cout << "Charges (before->after): " << chargeBefore << " -> " << chargeAfter << std::endl;
    std::cout << "Charge loss: " << (1 - chargeAfter/chargeBefore)*100 << " %" << std::endl << std::endl;
  }

  return true;

}

// *******************************************************************************************************
//! Reweight CMSSW clusters to look like clusters corresponding to Pixelav Templates.     
//! \param       id_in - (input) identifier of the template corresponding to the input events                                 
//! \param    id_rewgt - (input) identifier of the template corresponding to the output events                                 
//! \param     cluster - (input/output) boost multi_array container of 7x21 array of pixel signals, 
//!                       origin of local coords (0,0) at center of pixel cluster[3][10].                      
//! returns 0 if everything is OK, 1 if angles are outside template coverage (cluster is probably still
//! usable, > 1 if something is wrong (no reweight done).
// *******************************************************************************************************
int SiPixelDigitizerAlgorithm::PixelTempRewgt2D(int id_in, int id_rewgt, array_2d& cluster)
{
  // Local variables 
  int i, j, k, l, kclose;
  int nclusx, nclusy, success;
  float xsize, ysize, q50i, q100i, q50r, q10r, q100r, xhit2D, yhit2D, qclust, dist2, dmin2;
  float xy_in[BXM2][BYM2], xy_rewgt[BXM2][BYM2], xy_clust[TXSIZE][TYSIZE];
  int denx_clust[TXSIZE][TYSIZE], deny_clust[TXSIZE][TYSIZE];
  int goodWeightsUsed, nearbyWeightsUsed, noWeightsUsed;
  float cotalpha, cotbeta; 
  // success = 0 is returned if everthing is OK
  success = 0;
    
  // Copy the array to remember original charges
  array_2d clust(cluster);

  // Take the pixel dimensions from the 2D template
  templ2D.getid(id_in);
  xsize = templ2D.xsize();
  ysize = templ2D.ysize();
  
  // Calculate the track angles

  if (std::abs(track[5]) > 0.f){
    cotalpha = track[3]/track[5]; //if track[5] (direction in z) is 0 the hit is not processed by re-weighting
    cotbeta = track[4]/track[5];
  } else {
    LogDebug ("Pixel Digitizer") << "Reweighting angle is not good!" << std::endl;
    return 9; //returned value here indicates that no reweighting was done in this case
  }

  // The 2-D templates are defined on a shifted coordinate system wrt the 1D templates
  if(ydouble[0]) {
    yhit2D = track[1] - cotbeta*track[2] + ysize;
  } else {
    yhit2D = track[1] - cotbeta*track[2] + 0.5f*ysize;
  }
  if(xdouble[0]) {
    xhit2D = track[0] - cotalpha*track[2] + xsize;
  } else {
    xhit2D = track[0] - cotalpha*track[2] + 0.5f*xsize;
  }

  // Zero the input and output templates
  for(i=0; i<BYM2; ++i) {
    for(j=0; j<BXM2; ++j) {
      xy_in[j][i] = 0.f;
      xy_rewgt[j][i] = 0.f;
    }
  }

  // Next, interpolate the CMSSW template needed to analyze this cluster     

  if(!templ2D.xytemp(id_in, cotalpha, cotbeta, xhit2D, yhit2D, ydouble, xdouble, xy_in)) {success = 1;}
  if(success != 0){
#ifdef TP_DEBUG 
    LogDebug("Pixel Digitizer") << "No matching template found" << std::endl;
#endif
    return 2;
  }

  if(PrintTemplates){
    std::cout << "Template unirrad: " << std::endl;
    printCluster(xy_in);
  }

  q50i = templ2D.s50();
  //q50i = 0;
  q100i = 2.f*q50i;

  // Check that the cluster container is a 13x21 matrix
  
  if(cluster.num_dimensions() != 2) {
    LogWarning ("Pixel Digitizer") << "Cluster is not 2-dimensional. Return." << std::endl;
    return 3;
  }
  nclusx = (int)cluster.shape()[0];
  nclusy = (int)cluster.shape()[1];
  if(nclusx != TXSIZE || xdouble.size() != TXSIZE) {
    LogWarning ("Pixel Digitizer") << "Sizes in x do not match: nclusx=" << nclusx << "  xdoubleSize=" << xdouble.size() << "  TXSIZE=" << TXSIZE << ". Return." << std::endl;
    return 4;
  }
  if(nclusy != TYSIZE || ydouble.size() != TYSIZE) {
    LogWarning ("Pixel Digitizer") << "Sizes in y do not match. Return." << std::endl;
    return 5;
  }
  
  // Sum initial charge in the cluster   
  
  qclust = 0.f;
  for(i=0; i<TYSIZE; ++i) {
    for(j=0; j<TXSIZE; ++j) {
      xy_clust[j][i] = 0.f;
      denx_clust[j][i] = 0;
      deny_clust[j][i] = 0;
      if(cluster[j][i] > q100i) {
	qclust += cluster[j][i];
      }
    }
  }
  
  // Next, interpolate the physical output template needed to reweight     

  if(!templ2D.xytemp(id_rewgt, cotalpha, cotbeta, xhit2D, yhit2D, ydouble, xdouble, xy_rewgt)) {success = 1;}

  if(PrintTemplates){
    std::cout << "Template irrad: " << std::endl;  
    printCluster(xy_rewgt);
  }
    
  q50r = templ2D.s50();
  q100r = 2.f*q50r;
  q10r = 0.2f*q50r;
  
  // Find all non-zero denominator pixels in the input template and generate "inside" weights
     
  int ntpix = 0;
  int ncpix = 0;
  std::vector<int> ytclust;
  std::vector<int> xtclust;
  std::vector<int> ycclust;
  std::vector<int> xcclust;
  qclust = 0.f;
  for(i=0; i<TYSIZE; ++i) {
    for(j=0; j<TXSIZE; ++j) {
      if(xy_in[j+1][i+1] > q100i) {
	++ntpix;
	ytclust.push_back(i);
	xtclust.push_back(j);
	xy_clust[j][i] = xy_rewgt[j+1][i+1]/xy_in[j+1][i+1];
	denx_clust[j][i] = j;
	deny_clust[j][i] = i;
      }
    }
  }
  
  // Find all non-zero numerator pixels not matched to denominator in the output template and generate "inside" weights
    
  for(i=0; i<TYSIZE; ++i) {
    for(j=0; j<TXSIZE; ++j) {
      if(xy_rewgt[j+1][i+1] > q10r && xy_clust[j][i] == 0.f && ntpix>0) {
	// Search for nearest denominator pixel
	dmin2 = 10000.f; kclose = 0;
	for(k=0; k<ntpix; ++k) {
	  dist2=(i-ytclust[k])*(i-ytclust[k])+0.44444f*(j-xtclust[k])*(j-xtclust[k]);
	  if(dist2 < dmin2) {
	    dmin2 = dist2;
	    kclose = k;
	  }
	}
	xy_clust[j][i] = xy_rewgt[j+1][i+1]/xy_in[xtclust[kclose]+1][ytclust[kclose]+1];
	denx_clust[j][i] = xtclust[kclose];
	deny_clust[j][i] = ytclust[kclose];
      }
    }
  }

  if(PrintTemplates){
    std::cout << "Weights:" << std::endl;
    printCluster(xy_clust);
  }
  
  
  
  // Do the reweighting
  goodWeightsUsed = 0;
  nearbyWeightsUsed = 0;
  noWeightsUsed = 0;

  for(i=0; i<TYSIZE; ++i) {
    for(j=0; j<TXSIZE; ++j) {
      if(xy_clust[j][i] > 0.f) {
	cluster[j][i] = xy_clust[j][i]*clust[denx_clust[j][i]][deny_clust[j][i]];
	if(cluster[j][i] > q100r) {
	  qclust += cluster[j][i];
	}
	if(cluster[j][i] > 0) {
	  goodWeightsUsed++;
	}
      } else {
	if(clust[j][i] > 0.f) {
	  ++ncpix;
	  ycclust.push_back(i);
	  xcclust.push_back(j);
	}
      }
    }
  }
     
  // Now reweight pixels outside of template footprint using closest weights

  if(ncpix > 0) {
    for(l=0; l<ncpix; ++l) {
      i=ycclust[l]; j=xcclust[l];
      dmin2 = 10000.f; kclose = 0;
      for(k=0; k<ntpix; ++k) {
	dist2=(i-ytclust[k])*(i-ytclust[k])+0.44444f*(j-xtclust[k])*(j-xtclust[k]);
	if(dist2 < dmin2) {
	  dmin2 = dist2;
	  kclose = k;
	}
      }
      if(dmin2 < 5.f) {
	nearbyWeightsUsed++;
	cluster[j][i] *= xy_clust[xtclust[kclose]][ytclust[kclose]];
	if(cluster[j][i] > q100r) {
	  qclust += cluster[j][i];
	}
      } else {
	noWeightsUsed++;
	cluster[j][i] = 0.f;
      }
    }
  }

  return success;
} // PixelTempRewgt2D 

void SiPixelDigitizerAlgorithm::printCluster(array_2d& cluster)
{
  for(int col = 0; col < TYSIZE; ++col) {
    for(int row = 0; row < TXSIZE; ++row) {
      std::cout << std::setw(10) << std::setprecision(0) << std::fixed;
      std::cout << cluster[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}

void SiPixelDigitizerAlgorithm::printCluster(float arr[BXM2][BYM2])
{
  for(int col = 0; col < BYM2; ++col) {
    for(int row = 0; row < BXM2; ++row) {
      std::cout << std::setw(10) << std::setprecision(0) << std::fixed;
      std::cout << arr[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}

void SiPixelDigitizerAlgorithm::printCluster(float arr[TXSIZE][TYSIZE])
{
  for(int col = 0; col < TYSIZE; ++col) {
    for(int row = 0; row < TXSIZE; ++row) {
      std::cout << std::setw(10) << std::fixed;
      std::cout << arr[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}
