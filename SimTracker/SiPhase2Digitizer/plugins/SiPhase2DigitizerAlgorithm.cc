#include <iostream>
#include <math.h>

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimTracker/SiPhase2Digitizer/plugins/SiPhase2DigitizerAlgorithm.h"

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


void SiPhase2DigitizerAlgorithm::init(const edm::EventSetup& es) {
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

SiPhase2DigitizerAlgorithm::SiPhase2DigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng) :

  _signal(),
  makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", true)),
  use_ineff_from_db_(conf.getParameter<bool>("Inefficiency_DB")),
  use_module_killing_(conf.getParameter<bool>("KillModules")), // boolean to kill or not modules
  use_deadmodule_DB_(conf.getParameter<bool>("DeadModules_DB")), // boolean to access dead modules from DB
  use_LorentzAngle_DB_(conf.getParameter<bool>("LorentzAngle_DB")), // boolean to access Lorentz angle from DB

  DeadModules(use_deadmodule_DB_ ? Parameters() : conf.getParameter<Parameters>("DeadModules")), // get dead module from cfg file

  // Common pixel parameters
  // These are parameters which are not likely to be changed
  GeVperElectron(3.61E-09), // 1 electron(3.61eV, 1keV(277e, mod 9/06 d.k.
  alpha2Order(conf.getParameter<bool>("Alpha2Order")),      // switch on/off of E.B effect
  addXtalk(conf.getParameter<bool>("AddXTalk")), 
  interstripCoupling(conf.getParameter<double>("InterstripCoupling")), // Interstrip Coupling
 
  Sigma0(conf.getParameter<double>("SigmaZero")),           // Charge diffusion constant 7->3.7
  SigmaCoeff(conf.getParameter<double>("SigmaCoeff")),      // delta in the diffusion across the strip pitch 
                              // (set between 0 to 0.9,  0-->flat Sigma0, 1-->Sigma_min=0 & Sigma_max=2*Sigma0
   //D.B.: Dist300 replaced by moduleThickness, may not work with partially depleted sensors but works otherwise
   //Dist300(0.0300),                                          //   normalized to 300micron Silicon

  ClusterWidth(conf.getParameter<double>("ClusterWidth")),  // Charge integration spread on the collection plane

  doDigitalReadout(conf.getParameter<bool>("DigitalReadout")),         //  Flag to decide analog or digital readout
  // ADC calibration 1adc count(135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc](2[adc/kev]
  // Be carefull, this parameter is also used in SiPixelDet.cc to
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC(conf.getParameter<double>("ElectronPerAdc")),

  // ADC saturation value, 255(8bit adc.
  theAdcFullScale(conf.getParameter<int>("AdcFullScale")),
  
  // Noise in electrons:
  // Pixel cell noise, relevant for generating noisy pixels
  theNoiseInElectrons(conf.getParameter<double>("NoiseInElectrons")),

  // Fill readout noise, including all readout chain, relevant for smearing
  //theReadoutNoise(conf.getUntrackedParameter<double>("ReadoutNoiseInElec",500.)),
  theReadoutNoise(conf.getParameter<double>("ReadoutNoiseInElec")),

  // Threshold in units of noise:
  // thePixelThreshold(conf.getParameter<double>("ThresholdInNoiseUnits")),
  // Pixel threshold in electron units.
  theThresholdInE_Endcap(conf.getParameter<double>("ThresholdInElectrons_Endcap")),
  theThresholdInE_Barrel(conf.getParameter<double>("ThresholdInElectrons_Barrel")),

  // Add threshold gaussian smearing:
  theThresholdSmearing_Endcap(conf.getParameter<double>("ThresholdSmearing_Endcap")),
  theThresholdSmearing_Barrel(conf.getParameter<double>("ThresholdSmearing_Barrel")),

  //theTofCut 12.5, cut in particle TOD +/- 12.5ns
  //theTofCut(conf.getUntrackedParameter<double>("TofCut",12.5)),
  theTofLowerCut(conf.getParameter<double>("TofLowerCut")),
  theTofUpperCut(conf.getParameter<double>("TofUpperCut")),

  // Get the Lorentz angle from the cfg file:
  tanLorentzAnglePerTesla_Endcap(use_LorentzAngle_DB_ ? 0.0 : conf.getParameter<double>("TanLorentzAnglePerTesla_Endcap")),
  tanLorentzAnglePerTesla_Barrel(use_LorentzAngle_DB_ ? 0.0 : conf.getParameter<double>("TanLorentzAnglePerTesla_Barrel")),

  // Add noise
  addNoise(conf.getParameter<bool>("AddNoise")),

  // Add noisy pixels
  addNoisyPixels(conf.getParameter<bool>("AddNoisyPixels")),

  // Fluctuate charge in track subsegments
  fluctuateCharge(conf.getUntrackedParameter<bool>("FluctuateCharge",true)),

  // Control the pixel inefficiency
  AddPixelInefficiency(conf.getParameter<bool>("AddInefficiency")),

  // Add threshold gaussian smearing:
  addThresholdSmearing(conf.getParameter<bool>("AddThresholdSmearing")),

  // Add some pseudo-red damage
  pseudoRadDamage(conf.exists("PseudoRadDamage")?conf.getParameter<double>("PseudoRadDamage"):double(0.0)),
  pseudoRadDamageRadius(conf.exists("PseudoRadDamageRadius")?conf.getParameter<double>("PseudoRadDamageRadius"):double(0.0)),

  // delta cutoff in MeV, has to be same as in OSCAR(0.030/cmsim=1.0 MeV
  //tMax(0.030), // In MeV.
  //tMax(conf.getUntrackedParameter<double>("DeltaProductionCut",0.030)),
  tMax(conf.getParameter<double>("DeltaProductionCut")),

  fluctuate(fluctuateCharge ? new SiG4UniversalFluctuation(eng) : 0),
  theNoiser(addNoise ? new GaussianTailNoiseGenerator(eng) : 0),
  theSiPixelGainCalibrationService_(use_ineff_from_db_ ? new SiPixelGainCalibrationOfflineSimService(conf) : 0),
  subdetEfficiencies_(conf),
  flatDistribution_((addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) ? new CLHEP::RandFlat(eng, 0., 1.) : 0),
  gaussDistribution_((addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) ? new CLHEP::RandGaussQ(eng, 0., theReadoutNoise) : 0),
  // Threshold smearing with gaussian distribution:
  smearedThreshold_Endcap_(addThresholdSmearing ? new CLHEP::RandGaussQ(eng, theThresholdInE_Endcap , theThresholdSmearing_Endcap) : 0),
  smearedThreshold_Barrel_(addThresholdSmearing ? new CLHEP::RandGaussQ(eng, theThresholdInE_Barrel , theThresholdSmearing_Barrel) : 0)
{
  LogInfo ("PixelDigitizer ") <<"SiPhase2DigitizerAlgorithm constructed"
			      <<"Configuration parameters:"
			      << "Threshold/Gain = "
			      << "threshold in electron Endcap = "
			      << theThresholdInE_Endcap
			      << "threshold in electron Barrel = "
			      << theThresholdInE_Barrel
			      <<" " << theElectronPerADC << " " << theAdcFullScale
			      << " The delta cut-off is set to " << tMax
			      << " pix-inefficiency "<<AddPixelInefficiency;

  std::cout << "use_ineff_from_db_ " << use_ineff_from_db_
            << " use_module_killing_ " << use_module_killing_
            << " AddPixelInefficiency " << AddPixelInefficiency
            << " use_deadmodule_DB_ " << use_deadmodule_DB_
            << std::endl;  

}


//=========================================================================
SiPhase2DigitizerAlgorithm::~SiPhase2DigitizerAlgorithm() {
  LogDebug ("PixelDigitizer")<<"SiPhase2DigitizerAlgorithm deleted";
}

SiPhase2DigitizerAlgorithm::SubdetEfficiencies::SubdetEfficiencies(const edm::ParameterSet& conf) {
  barrel_efficiencies = conf.getParameter< std::vector<double> >("EfficiencyFactors_Barrel");
  endcap_efficiencies = conf.getParameter< std::vector<double> >("EfficiencyFactors_Endcap");
}

//=========================================================================
void SiPhase2DigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
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
      if (  ((*ssbegin).tof() - pixdet->surface().toGlobal((*ssbegin).localPosition()).mag()/30.)>= theTofLowerCut &&
	    ((*ssbegin).tof()- pixdet->surface().toGlobal((*ssbegin).localPosition()).mag()/30.) <= theTofUpperCut ) {
	primary_ionization(*ssbegin, ionization_points); // fills _ionization_points
	drift (*ssbegin, pixdet, bfield, ionization_points, collection_points);  // transforms _ionization_points to collection_points
	// compute induced signal on readout elements and add to _signal
	induce_signal(*ssbegin, pixdet, collection_points); // *ihit needed only for SimHit<-->Digi link
      } //  end if
    } // end for
}

//============================================================================
void SiPhase2DigitizerAlgorithm::digitize(const PixelGeomDetUnit* pixdet,
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

  float theThresholdInE = 0.;

  if(theNoiseInElectrons>0.){
    if(Sub_detid == PixelSubdetector::PixelBarrel){ // Barrel modules
      if(addThresholdSmearing) theThresholdInE = smearedThreshold_Barrel_->fire(); // gaussian smearing
      else theThresholdInE = theThresholdInE_Barrel; // no smearing
    } else { // Forward disks modules
      if(addThresholdSmearing) theThresholdInE = smearedThreshold_Endcap_->fire(); // gaussian smearing
      else theThresholdInE = theThresholdInE_Endcap; // no smearing
    }
  }

#ifdef TP_DEBUG
    // full detector thickness
   float moduleThickness = pixdet->specificSurface().bounds().thickness();
    LogDebug ("PixelDigitizer")
      << " PixelDigitizer "
      << numColumns << " " << numRows << " " << moduleThickness;
#endif

    if(addNoise) add_noise(pixdet, theThresholdInE/theNoiseInElectrons);  // generate noise

    // Do only if needed

    if((AddPixelInefficiency) && (theSignal.size()>0)) {
      if (use_ineff_from_db_) pixel_inefficiency_db(detID);
      else                    pixel_inefficiency(subdetEfficiencies_, pixdet, tTopo);
    }

    if(use_module_killing_) {
      if (use_deadmodule_DB_) {  // remove dead modules using DB
        module_killing_DB(detID);
      } else { // remove dead modules using the list in cfg file
        module_killing_conf(detID);
      }
    }

    make_digis(theThresholdInE, detID, digis, simlinks, tTopo);

#ifdef TP_DEBUG
  LogDebug ("PixelDigitizer") << "[SiPhase2DigitizerAlgorithm] converted " << digis.size() << " PixelDigis in DetUnit" << detID;
#endif
}

//***********************************************************************/
// Generate primary ionization along the track segment.
// Divide the track into small sub-segments
void SiPhase2DigitizerAlgorithm::primary_ionization(const PSimHit& hit, std::vector<EnergyDepositUnit>& ionization_points) const {

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
void SiPhase2DigitizerAlgorithm::fluctuateEloss(int pid, float particleMomentum,
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
void SiPhase2DigitizerAlgorithm::drift(const PSimHit& hit,
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
  float stripPitch     = pixdet->specificTopology().pitch().first;
 
 
  //  int rowsPerRoc = pixdet->specificTopology().rowsperroc();	//D.B.:debug
  //  int colsPerRoc = pixdet->specificTopology().colsperroc();	//D.B.:debug
  //  int nColumns = pixdet->specificTopology().ncolumns();	//D.B.:debug
  //  int nRows = pixdet->specificTopology().nrows();	//D.B.:debug
 
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
    Sigma = sqrt(DriftLength/moduleThickness) * (Sigma0 * moduleThickness/0.0300); //Sigma0=0.00037 is for 300um thickness (make sure moduleThickness is in [cm])
    //D.B.: sigmaCoeff=0 means no modulation
    if (SigmaCoeff) Sigma *= (SigmaCoeff*cos(SegX*M_PI/stripPitch)*cos(SegX*M_PI/stripPitch)+1); 
                                                     //NB: divided by 4 to get a periodicity of stripPitch

    // Project the diffusion sigma on the collection plane
    Sigma_x = Sigma / CosLorenzAngleX ;
    Sigma_y = Sigma / CosLorenzAngleY ;
    //    if (rowsPerRoc==143) {//D.B.:debug
    //    std::cout<<" stripPitch="<<stripPitch<<" rowsPerRoc="<<rowsPerRoc<<" colsPerRoc="<<colsPerRoc<<" nColumns="<<nColumns<<" nRows"<<nRows<<std::endl;//D.B.: for debug
    //    }

    // Insert a charge loss due to Rad Damage here
    float energyOnCollector = ionization_points[i].energy(); // The energy that reaches the collector
    // pseudoRadDamage
    if (pseudoRadDamage>=0.001){
      float moduleRadius = pixdet->surface().position().perp();
      if (moduleRadius<=pseudoRadDamageRadius){
	float kValue = pseudoRadDamage /(moduleRadius*moduleRadius);
	//std::cout <<"\nmoduleRadius= "<<moduleRadius<<" WITH K="<<kValue <<" wich scales the energy by "<<
	//            exp( -1*kValue*DriftDistance/moduleThickness )<<"\n" ;
	energyOnCollector = energyOnCollector * exp( -1*kValue*DriftDistance/moduleThickness );
      }
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
void SiPhase2DigitizerAlgorithm::induce_signal(const PSimHit& hit,
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
   int iseg = 0; 
   float ESum = 0.0;
   // Iterate over collection points on the collection plane
   for ( std::vector<SignalPoint>::const_iterator i=collection_points.begin();
	 i != collection_points.end(); ++i) {
     iseg++;
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

    // Now integarte strips in y
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
	ESum += ChargeFraction;  
	
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
void SiPhase2DigitizerAlgorithm::make_digis(float theThresholdInE,
                                           uint32_t detID,
                                           std::vector<PixelDigi>& digis,
                                           std::vector<PixelDigiSimLink>& simlinks,
					   const TrackerTopology *tTopo) const  {

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " make digis "<<" "
			       << " pixel threshold Endcap" << theThresholdInE_Endcap << " "
                               << " pixel threshold Barrel" << theThresholdInE_Barrel << " "
			       << " List pixels passing threshold ";
#endif

  // Loop over hit pixels

  signalMaps::const_iterator it = _signal.find(detID);
  if (it == _signal.end()) {
    return;
  }

  const signal_map_type& theSignal = (*it).second;

  for (signal_map_const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {

    float signalInElectrons = (*i).second.ampl() ;   // signal in electrons

    // Do only for pixels above threshold

    if( signalInElectrons >= theThresholdInE) { // check threshold

      int chan =  (*i).first;  // channel number
      std::pair<int,int> ip = PixelDigi::channelToPixel(chan);

      int adc = 0;

#ifdef TP_DEBUG
      LogDebug ("Pixel Digitizer")
	<< (*i).first << " " << (*i).second << " " << signalInElectrons
	<< " " << adc << ip.first << " " << ip.second ;
#endif
      if (doDigitalReadout) adc = theAdcFullScale;
      else adc = int( signalInElectrons / theElectronPerADC );
 
      // Load digis
      digis.emplace_back(ip.first, ip.second, adc);

      if (makeDigiSimLinks_ && (*i).second.hitInfo()!=0) {
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
void SiPhase2DigitizerAlgorithm::add_noise(const PixelGeomDetUnit* pixdet,
                                          float thePixelThreshold) {
#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter add_noise " << theNoiseInElectrons;
#endif

  uint32_t detID= pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
   const PixelTopology* topol=&pixdet->specificTopology();
   int numColumns = topol->ncolumns();  // det module number of cols&rows
   int numRows = topol->nrows();

  // First add Readout noise to hit cells
  for ( signal_map_iterator i = theSignal.begin(); i != theSignal.end(); i++) {
    float noise  = gaussDistribution_->fire() ;
    if(((*i).second.ampl() + noise) < 0. ) {
      (*i).second.set(0);}
    else{
      (*i).second += noise;
    }
  }
  
  if (addXtalk) {
    signal_map_type signalNew;
    for (signal_map_iterator i = theSignal.begin(); i != theSignal.end(); i++) {
      float signalInElectrons = (*i).second.ampl();   // signal in electrons
      std::pair<int,int> hitChan = PixelDigi::channelToPixel((*i).first);
      
      float signalInElectrons_Xtalk = signalInElectrons * interstripCoupling;     
      
      if (hitChan.first != 0) {	
	std::pair<int,int> XtalkPrev = std::pair<int,int>(hitChan.first-1, hitChan.second);
	int chanXtalkPrev = PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
	signalNew.insert(std::pair<int,Amplitude>(chanXtalkPrev, Amplitude (signalInElectrons_Xtalk, -1.0)));
      }
      if (hitChan.first < (numRows-1)) {
	std::pair<int,int> XtalkNext = std::pair<int,int>(hitChan.first+1, hitChan.second);
	int chanXtalkNext = PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
	signalNew.insert(std::pair<int,Amplitude>(chanXtalkNext, Amplitude (signalInElectrons_Xtalk, -1.0)));
      }
    }
    for (signal_map_iterator l = signalNew.begin(); l != signalNew.end(); l++) {
      int chan = l->first;
      signal_map_iterator iter = theSignal.find(chan);
      if (iter != theSignal.end()) {
	theSignal[chan] += l->second.ampl();
      }  else theSignal.insert(std::pair<int,Amplitude>(chan, Amplitude(l->second.ampl(),-1.0)));
    } 
#if 0
    for (signal_map_iterator l = theSignal.begin(); l != theSignal.end(); l++) {
      std::pair<int,int> hc = PixelDigi::channelToPixel((*l).first);
      std::cout << (*l).first << " " << hc.first << "  " << hc.second << "  " << (*l).second << "  " << std::endl;
    }
#endif    
  } 
  if (!addNoisyPixels)  // Option to skip noise in non-hit pixels
    return;

  // Add noise on non-hit pixels
  // Use here the pixel noise
  int numberOfPixels = numRows * numColumns;
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
    << theThresholdInE_Endcap << "  " << theThresholdInE_Barrel <<" "<< numberOfPixels<<" "
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

    if (theSignal[chan] == 0) {
      //      float noise = float( (*mapI).second );
      int noise=int( (*mapI).second );
      theSignal[chan] = Amplitude (noise, -1.);
    }
  }
}

/***********************************************************************/

// Simulate the readout inefficiencies.
// Delete a selected number of single pixels, dcols and rocs.
void SiPhase2DigitizerAlgorithm::pixel_inefficiency(const SubdetEfficiencies& eff,
			                           const PixelGeomDetUnit* pixdet,
						   const TrackerTopology *tTopo) {

  uint32_t detID= pixdet->geographicalId().rawId();

  signal_map_type& theSignal = _signal[detID];

  // Predefined efficiencies
  float subdetEfficiency  = 1.0;

  // setup the chip indices conversion
  unsigned int Subid=DetId(detID).subdetId();
  if    (Subid==  PixelSubdetector::PixelBarrel){// barrel layers
    unsigned int layerIndex=tTopo->pxbLayer(detID);
    if (layerIndex-1 < eff.barrel_efficiencies.size() )   subdetEfficiency  = eff.barrel_efficiencies[layerIndex-1];
    else std::cout << (layerIndex-1) << " " << eff.barrel_efficiencies.size() << std::endl;
  } else {                // forward disks
    unsigned int diskIndex = 2 * tTopo->pxfDisk(detID) - tTopo->pxfSide(detID);
    if (diskIndex-1 < eff.endcap_efficiencies.size()) subdetEfficiency  = eff.endcap_efficiencies[diskIndex-1];
    else std::cout << diskIndex-1 << " " << eff.barrel_efficiencies.size() << std::endl;
  } // if barrel/forward

#ifdef TP_DEBUG
  LogDebug ("Pixel Digitizer") << " enter pixel_inefficiency " << subdetEfficiency;
#endif

  // Now loop again over pixels to kill some of them.
  // Loop over hits, amplitude in electrons, channel = coded row,col
  for(signal_map_iterator i = theSignal.begin();i != theSignal.end(); ++i) {
    float rand  = flatDistribution_->fire();
    if( rand>subdetEfficiency ) {
      // make amplitude =0
      i->second.set(0.); // reset amplitude,
    } // end if
  } 
} 
//***********************************************************************
//******************************************************************************

// Set the drift direction accoring to the Bfield in local det-unit frame
// Works for both barrel and forward pixels.
// Replace the sign convention to fit M.Swartz's formulaes.
// Configurations for barrel and foward pixels possess different tanLorentzAngleperTesla
// parameter value

LocalVector SiPhase2DigitizerAlgorithm::DriftDirection(const PixelGeomDetUnit* pixdet,
                                                      const GlobalVector& bfield,
                                                      const DetId& detId) const {
  Frame detFrame(pixdet->surface().position(),pixdet->surface().rotation());
  LocalVector Bfield=detFrame.toLocal(bfield);
  float alpha2_Endcap;
  float alpha2_Barrel;
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
      alpha2_Endcap = tanLorentzAnglePerTesla_Endcap*tanLorentzAnglePerTesla_Endcap;
      alpha2_Barrel = tanLorentzAnglePerTesla_Barrel*tanLorentzAnglePerTesla_Barrel;
    }else {
      alpha2_Endcap = 0.0;
      alpha2_Barrel = 0.0;
    }
    
    if(Sub_detid == PixelSubdetector::PixelBarrel){// barrel layers
      dir_x = -( tanLorentzAnglePerTesla_Barrel * Bfield.y() + alpha2_Barrel* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_Barrel * Bfield.x() - alpha2_Barrel* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_Barrel* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_Barrel* Bfield.z()*Bfield.z() );

    } else {// forward disks
      dir_x = -( tanLorentzAnglePerTesla_Endcap * Bfield.y() + alpha2_Endcap* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_Endcap * Bfield.x() - alpha2_Endcap* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_Endcap* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_Endcap* Bfield.z()*Bfield.z() );
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

void SiPhase2DigitizerAlgorithm::pixel_inefficiency_db(uint32_t detID) {
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

void SiPhase2DigitizerAlgorithm::module_killing_conf(uint32_t detID) {
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
void SiPhase2DigitizerAlgorithm::module_killing_DB(uint32_t detID) {
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

