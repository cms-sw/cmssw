#include<typeinfo>
#include <iostream>
#include <cmath>

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

//#include "PixelIndices.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

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

Phase2TrackerDigitizerAlgorithm::Phase2TrackerDigitizerAlgorithm(const edm::ParameterSet& conf_common, 
								 const edm::ParameterSet& conf_specific):
  _signal(),
  makeDigiSimLinks_(conf_common.getUntrackedParameter<bool>("makeDigiSimLinks", true)),
  use_ineff_from_db_(conf_specific.getParameter<bool>("Inefficiency_DB")),
  use_module_killing_(conf_specific.getParameter<bool>("KillModules")), // boolean to kill or not modules
  use_deadmodule_DB_(conf_specific.getParameter<bool>("DeadModules_DB")), // boolean to access dead modules from DB
  use_LorentzAngle_DB_(conf_specific.getParameter<bool>("LorentzAngle_DB")), // boolean to access Lorentz angle from DB
  
  DeadModules(use_deadmodule_DB_ ? Parameters() : conf_specific.getParameter<Parameters>("DeadModules")), // get dead module from cfg file

  // Common pixel parameters
  // These are parameters which are not likely to be changed
  GeVperElectron(3.61E-09), // 1 electron(3.61eV, 1keV(277e, mod 9/06 d.k.
  alpha2Order(conf_specific.getParameter<bool>("Alpha2Order")),      // switch on/off of E.B effect
  addXtalk(conf_specific.getParameter<bool>("AddXTalk")), 
  interstripCoupling(conf_specific.getParameter<double>("InterstripCoupling")), // Interstrip Coupling
  
  Sigma0(conf_specific.getParameter<double>("SigmaZero")),           // Charge diffusion constant 7->3.7
  SigmaCoeff(conf_specific.getParameter<double>("SigmaCoeff")),      // delta in the diffusion across the strip pitch 
  // (set between 0 to 0.9,  0-->flat Sigma0, 1-->Sigma_min=0 & Sigma_max=2*Sigma0
  // D.B.: Dist300 replaced by moduleThickness, may not work with partially depleted sensors but works otherwise
  // Dist300(0.0300),                                          //   normalized to 300micron Silicon

  ClusterWidth(conf_specific.getParameter<double>("ClusterWidth")),  // Charge integration spread on the collection plane

  // Allowed modes of readout which has following values :
  // 0          ---> Digital or binary readout 
  // -1         ---> Analog readout, current digitizer (Inner Pixel) (TDR version) with no threshold subtraction
  // Analog readout with dual slope with the "second" slope being 1/2^(n-1) and threshold subtraction (n = 1, 2, 3,4)
  thePhase2ReadoutMode(conf_specific.getParameter<int>("Phase2ReadoutMode")), 

  // ADC calibration 1adc count(135e.
  // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc](2[adc/kev]
  // Be careful, this parameter is also used in SiPixelDet.cc to
  // calculate the noise in adc counts from noise in electrons.
  // Both defaults should be the same.
  theElectronPerADC(conf_specific.getParameter<double>("ElectronPerAdc")),

  // ADC saturation value, 255(8bit adc.
  theAdcFullScale(conf_specific.getParameter<int>("AdcFullScale")),
  
  // Noise in electrons:
  // Pixel cell noise, relevant for generating noisy pixels
  theNoiseInElectrons(conf_specific.getParameter<double>("NoiseInElectrons")),

  // Fill readout noise, including all readout chain, relevant for smearing
  theReadoutNoise(conf_specific.getParameter<double>("ReadoutNoiseInElec")),

  // Threshold in units of noise:
  // thePixelThreshold(conf.getParameter<double>("ThresholdInNoiseUnits")),
  // Pixel threshold in electron units.
  theThresholdInE_Endcap(conf_specific.getParameter<double>("ThresholdInElectrons_Endcap")),
  theThresholdInE_Barrel(conf_specific.getParameter<double>("ThresholdInElectrons_Barrel")),

  // Add threshold gaussian smearing:
  theThresholdSmearing_Endcap(conf_specific.getParameter<double>("ThresholdSmearing_Endcap")),
  theThresholdSmearing_Barrel(conf_specific.getParameter<double>("ThresholdSmearing_Barrel")),
  
  // Add HIP Threshold in electron units. 
  theHIPThresholdInE_Endcap(conf_specific.getParameter<double>("HIPThresholdInElectrons_Endcap")),
  theHIPThresholdInE_Barrel(conf_specific.getParameter<double>("HIPThresholdInElectrons_Barrel")),

  // theTofCut 12.5, cut in particle TOD +/- 12.5ns
  theTofLowerCut(conf_specific.getParameter<double>("TofLowerCut")),
  theTofUpperCut(conf_specific.getParameter<double>("TofUpperCut")),

  // Get the Lorentz angle from the cfg file:
  tanLorentzAnglePerTesla_Endcap(use_LorentzAngle_DB_ ? 0.0 : conf_specific.getParameter<double>("TanLorentzAnglePerTesla_Endcap")),
  tanLorentzAnglePerTesla_Barrel(use_LorentzAngle_DB_ ? 0.0 : conf_specific.getParameter<double>("TanLorentzAnglePerTesla_Barrel")),

  // Add noise
  addNoise(conf_specific.getParameter<bool>("AddNoise")),

  // Add noisy pixels
  addNoisyPixels(conf_specific.getParameter<bool>("AddNoisyPixels")),

  // Fluctuate charge in track subsegments
  fluctuateCharge(conf_specific.getUntrackedParameter<bool>("FluctuateCharge",true)),

  // Control the pixel inefficiency
  AddPixelInefficiency(conf_specific.getParameter<bool>("AddInefficiency")),

  // Add threshold gaussian smearing:
  addThresholdSmearing(conf_specific.getParameter<bool>("AddThresholdSmearing")),

  // Add some pseudo-red damage
  pseudoRadDamage(conf_specific.exists("PseudoRadDamage")?conf_specific.getParameter<double>("PseudoRadDamage"):double(0.0)),
  pseudoRadDamageRadius(conf_specific.exists("PseudoRadDamageRadius")?conf_specific.getParameter<double>("PseudoRadDamageRadius"):double(0.0)),

  // delta cutoff in MeV, has to be same as in OSCAR(0.030/cmsim=1.0 MeV
  // tMax(0.030), // In MeV.
  // tMax(conf.getUntrackedParameter<double>("DeltaProductionCut",0.030)),
  tMax(conf_common.getParameter<double>("DeltaProductionCut")),

  badPixels(conf_specific.getParameter<std::vector<edm::ParameterSet> >("CellsToKill")),
  fluctuate(fluctuateCharge ? new SiG4UniversalFluctuation() : nullptr),
  theNoiser(addNoise ? new GaussianTailNoiseGenerator() : nullptr),
  theSiPixelGainCalibrationService_(use_ineff_from_db_ ? new SiPixelGainCalibrationOfflineSimService(conf_specific) : nullptr),
  subdetEfficiencies_(conf_specific)
{

  LogInfo("Phase2TrackerDigitizerAlgorithm") << "Phase2TrackerDigitizerAlgorithm constructed\n"
			    << "Configuration parameters:\n"
			    << "Threshold/Gain = "
			    << "threshold in electron Endcap = "
			    << theThresholdInE_Endcap
			    << "\nthreshold in electron Barrel = "
			    << theThresholdInE_Barrel
			    << " ElectronPerADC " << theElectronPerADC 
			    << " ADC Scale (in bits) " << theAdcFullScale
			    << " The delta cut-off is set to " << tMax
			    << " pix-inefficiency " << AddPixelInefficiency;
}

Phase2TrackerDigitizerAlgorithm::~Phase2TrackerDigitizerAlgorithm() {
  LogDebug("Phase2TrackerDigitizerAlgorithm") << "Phase2TrackerDigitizerAlgorithm deleted";
}

Phase2TrackerDigitizerAlgorithm::SubdetEfficiencies::SubdetEfficiencies(const edm::ParameterSet& conf) {
  barrel_efficiencies = conf.getParameter< std::vector<double> >("EfficiencyFactors_Barrel");
  endcap_efficiencies = conf.getParameter< std::vector<double> >("EfficiencyFactors_Endcap");
}
// =================================================================
//
// Generate primary ionization along the track segment.
// Divide the track into small sub-segments
//
// =================================================================
void Phase2TrackerDigitizerAlgorithm::primary_ionization(const PSimHit& hit, 
							 std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points) const {
  // Straight line approximation for trajectory inside active media
  const float SegmentLength = 0.0010; // in cm (10 microns)
  float energy;

  // Get the 3D segment direction vector
  LocalVector direction = hit.exitPoint() - hit.entryPoint();

  float eLoss = hit.energyLoss();  // Eloss in GeV
  float length = direction.mag();  // Track length in Silicon

  int NumberOfSegments = int (length / SegmentLength); // Number of segments
  if (NumberOfSegments < 1) NumberOfSegments = 1;
  LogDebug("Phase2TrackerDigitizerAlgorithm")
    << "enter primary_ionzation " << NumberOfSegments
    << " shift = "
    << hit.exitPoint().x() - hit.entryPoint().x() << " "
    << hit.exitPoint().y() - hit.entryPoint().y() << " "
    << hit.exitPoint().z() - hit.entryPoint().z() << " "
    << hit.particleType() 
    << " " << hit.pabs();

  std::vector<float> elossVector;  // Eloss vector
  elossVector.reserve(NumberOfSegments);
  if (fluctuateCharge) {
    int pid = hit.particleType();
    // int pid=211;  // assume it is a pion
    
    float momentum = hit.pabs();
    // Generate fluctuated charge points
    fluctuateEloss(pid, momentum, eLoss, length, NumberOfSegments, elossVector);
  }
  ionization_points.reserve(NumberOfSegments); // set size

  // loop over segments
  for (int i = 0; i != NumberOfSegments; ++i) {
    // Divide the segment into equal length subsegments
    Local3DPoint point = hit.entryPoint() + float((i+0.5)/NumberOfSegments) * direction;
    if (fluctuateCharge)
      energy = elossVector[i]/GeVperElectron; // Convert charge to elec.
    else
      energy = hit.energyLoss()/GeVperElectron/float(NumberOfSegments);

    DigitizerUtility::EnergyDepositUnit edu(energy, point); // define position,energy point
    ionization_points.push_back(edu); // save
    LogDebug("Phase2TrackerDigitizerAlgorithm")
      << i << " " << ionization_points[i].x() << " "
      << ionization_points[i].y() << " "
      << ionization_points[i].z() << " "
      << ionization_points[i].energy();
  }
}
//==============================================================================
//
// Fluctuate the charge comming from a small (10um) track segment.
// Use the G4 routine. For mip pions for the moment.
//
//==============================================================================
void Phase2TrackerDigitizerAlgorithm::fluctuateEloss(int pid, 
						     float particleMomentum,
						     float eloss, 
						     float length,
						     int NumberOfSegs,
						     std::vector<float> & elossVector) const {

  // Get dedx for this track
  //float dedx;
  //if( length > 0.) dedx = eloss/length;
  //else dedx = eloss;

  double particleMass = 139.6; // Mass in MeV, Assume pion
  pid = std::abs(pid);
  if (pid != 211) {       // Mass in MeV
    if (pid == 11)        particleMass = 0.511;
    else if (pid == 13)   particleMass = 105.7;
    else if (pid == 321)  particleMass = 493.7;
    else if (pid == 2212) particleMass = 938.3;
  }
  // What is the track segment length.
  float segmentLength = length/NumberOfSegs;

  // Generate charge fluctuations.
  float de = 0.;
  float sum = 0.;
  double segmentEloss = (1000. * eloss)/NumberOfSegs; //eloss in MeV
  for (int i = 0; i < NumberOfSegs; ++i) {
    //       material,*,   momentum,energy,*, *,  mass
    //myglandz_(14.,segmentLength,2.,2.,dedx,de,0.14);
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    double deltaCutoff = tMax; // the cutoff is sometimes redefined inside, so fix it.
    de = fluctuate->SampleFluctuations(static_cast<double>(particleMomentum*1000.),
				       particleMass, deltaCutoff,
				       static_cast<double>(segmentLength*10.),
				       segmentEloss,
                                       rengine_ )/1000.; //convert to GeV
    elossVector.push_back(de);
    sum += de;
  }
  if (sum > 0.) {  // If fluctuations give eloss>0.
    // Rescale to the same total eloss
    float ratio = eloss/sum;
    for (int ii = 0; ii < NumberOfSegs; ++ii) elossVector[ii] = ratio*elossVector[ii];
  } 
  else {  // If fluctuations gives 0 eloss
    float averageEloss = eloss/NumberOfSegs;
    for (int ii = 0; ii < NumberOfSegs; ++ii) elossVector[ii] = averageEloss;
  }

}

// ======================================================================
//
// Drift the charge segments to the sensor surface (collection plane)
// Include the effect of E-field and B-field
//
// =====================================================================
void Phase2TrackerDigitizerAlgorithm::drift(const PSimHit& hit,
					    const Phase2TrackerGeomDetUnit* pixdet,
					    const GlobalVector& bfield,
					    const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points,
					    std::vector<DigitizerUtility::SignalPoint>& collection_points) const {
  LogDebug("Phase2TrackerDigitizerAlgorithm") << "enter drift ";

  collection_points.resize(ionization_points.size()); // set size
  LocalVector driftDir = DriftDirection(pixdet, bfield, hit.detUnitId());  // get the charge drift direction
  if (driftDir.z() == 0.) {
    LogWarning("Phase2TrackerDigitizerAlgorithm") << " pxlx: drift in z is zero ";
    return;
  }

  float TanLorenzAngleX, 
        TanLorenzAngleY, 
        dir_z, 
        CosLorenzAngleX, 
        CosLorenzAngleY;
  if (alpha2Order) {
    TanLorenzAngleX = driftDir.x(); // tangen of Lorentz angle
    TanLorenzAngleY = driftDir.y();
    dir_z = driftDir.z(); // The z drift direction
    CosLorenzAngleX = 1./std::sqrt(1. + TanLorenzAngleX * TanLorenzAngleX); // cosine
    CosLorenzAngleY = 1./std::sqrt(1. + TanLorenzAngleY * TanLorenzAngleY); // cosine;
  } 
  else {
    TanLorenzAngleX = driftDir.x();
    TanLorenzAngleY = 0.; // force to 0, driftDir.y()/driftDir.z();
    dir_z = driftDir.z(); // The z drift direction
    CosLorenzAngleX = 1./std::sqrt(1. + TanLorenzAngleX * TanLorenzAngleX); // cosine to estimate the path length
    CosLorenzAngleY = 1.;
  }

  float moduleThickness = pixdet->specificSurface().bounds().thickness();
  float stripPitch     = pixdet->specificTopology().pitch().first;
 
  LogDebug("Phase2TrackerDigitizerAlgorithm")
    << " Lorentz Tan " << TanLorenzAngleX << " " << TanLorenzAngleY <<" "
    << CosLorenzAngleX << " " << CosLorenzAngleY << " "
    << moduleThickness*TanLorenzAngleX << " " << driftDir;

  float Sigma_x = 1.;  // Charge spread
  float Sigma_y = 1.;
  float DriftDistance; // Distance between charge generation and collection
  float DriftLength;   // Actual Drift Lentgh
  float Sigma;

  for (unsigned int i = 0; i != ionization_points.size(); ++i) {
    float SegX, SegY, SegZ; // position
    SegX = ionization_points[i].x();
    SegY = ionization_points[i].y();
    SegZ = ionization_points[i].z();

    // Distance from the collection plane
    // DriftDistance = (moduleThickness/2. + SegZ); // Drift to -z
    // Include explixitely the E drift direction (for CMS dir_z=-1)
    DriftDistance = moduleThickness/2. - (dir_z * SegZ); // Drift to -z

    if (DriftDistance < 0.)
      DriftDistance = 0.;
    else if (DriftDistance > moduleThickness)
      DriftDistance = moduleThickness;

    // Assume full depletion now, partial depletion will come later.
    float XDriftDueToMagField = DriftDistance * TanLorenzAngleX;
    float YDriftDueToMagField = DriftDistance * TanLorenzAngleY;

    // Shift cloud center
    float CloudCenterX = SegX + XDriftDueToMagField;
    float CloudCenterY = SegY + YDriftDueToMagField;

    // Calculate how long is the charge drift path
    DriftLength = std::sqrt(DriftDistance*DriftDistance +
			    XDriftDueToMagField*XDriftDueToMagField +
			    YDriftDueToMagField*YDriftDueToMagField);

    // What is the charge diffusion after this path
    // Sigma0=0.00037 is for 300um thickness (make sure moduleThickness is in [cm])    
    Sigma = std::sqrt(DriftLength/moduleThickness) * (Sigma0 * moduleThickness/0.0300); 
    // D.B.: sigmaCoeff=0 means no modulation
    if (SigmaCoeff) Sigma *= (SigmaCoeff * cos(SegX*M_PI/stripPitch) * cos(SegX*M_PI/stripPitch) + 1); 
    // NB: divided by 4 to get a periodicity of stripPitch

    // Project the diffusion sigma on the collection plane
    Sigma_x = Sigma / CosLorenzAngleX;
    Sigma_y = Sigma / CosLorenzAngleY;

    // Insert a charge loss due to Rad Damage here
    float energyOnCollector = ionization_points[i].energy(); // The energy that reaches the collector

    // pseudoRadDamage
    if (pseudoRadDamage >= 0.001) {
      float moduleRadius = pixdet->surface().position().perp();
      if (moduleRadius <= pseudoRadDamageRadius) {
	float kValue = pseudoRadDamage/(moduleRadius * moduleRadius);
	energyOnCollector = energyOnCollector * exp(-1 * kValue * DriftDistance/moduleThickness);
      }
    }
    LogDebug("Phase2TrackerDigitizerAlgorithm")
      << "Dift DistanceZ = " << DriftDistance << " module thickness = " << moduleThickness
      << " Start Energy = " << ionization_points[i].energy() << " Energy after loss= " << energyOnCollector;
    DigitizerUtility::SignalPoint sp(CloudCenterX, CloudCenterY, Sigma_x, Sigma_y, hit.tof(), energyOnCollector);
    // Load the Charge distribution parameters
    collection_points[i] = sp;
  }
}

// ====================================================================
//
// Induce the signal on the collection plane of the active sensor area.
void Phase2TrackerDigitizerAlgorithm::induce_signal(const PSimHit& hit,
						    const size_t hitIndex,
						    const unsigned int tofBin,
						    const Phase2TrackerGeomDetUnit* pixdet,
						    const std::vector<DigitizerUtility::SignalPoint>& collection_points) {

  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];

  LogDebug("Phase2TrackerDigitizerAlgorithm")
    << " enter induce_signal, "
    << topol->pitch().first << " " << topol->pitch().second; //OK

  // local map to store pixels hit by 1 Hit.
  using hit_map_type =  std::map<int, float, std::less<int> >;
  hit_map_type hit_signal;

  // map to store pixel integrals in the x and in the y directions
  std::map<int, float, std::less<int> > x,y;

  // Assign signals to readout channels and store sorted by channel number
  int iseg = 0; 
  float ESum = 0.0;

  // Iterate over collection points on the collection plane
  for (auto const & v : collection_points) {
    iseg++;
    float CloudCenterX = v.position().x(); // Charge position in x
    float CloudCenterY = v.position().y(); //                 in y
    float SigmaX = v.sigma_x();            // Charge spread in x
    float SigmaY = v.sigma_y();            //               in y
    float Charge = v.amplitude();          // Charge amplitude
    
    LogDebug("Phase2TrackerDigitizerAlgorithm")
      << " cloud " << v.position().x() << " " << v.position().y() << " "
      << v.sigma_x() << " " << v.sigma_y() << " " << v.amplitude();

    // Find the maximum cloud spread in 2D plane , assume 3*sigma
    float CloudRight = CloudCenterX + ClusterWidth * SigmaX;
    float CloudLeft  = CloudCenterX - ClusterWidth * SigmaX;
    float CloudUp    = CloudCenterY + ClusterWidth * SigmaY;
    float CloudDown  = CloudCenterY - ClusterWidth * SigmaY;

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
    
    int IPixRightUpX = int(floor( mp.x()));
    int IPixRightUpY = int(floor( mp.y()));

    LogDebug("Phase2TrackerDigitizerAlgorithm") << " right-up " << PointRightUp << " "
						<< mp.x() << " " << mp.y() << " "
						<< IPixRightUpX << " " << IPixRightUpY ;

    mp = topol->measurementPosition(PointLeftDown); // OK

    int IPixLeftDownX = int(floor( mp.x()));
    int IPixLeftDownY = int(floor( mp.y()));

    LogDebug("Phase2TrackerDigitizerAlgorithm") << " left-down " << PointLeftDown << " "
				<< mp.x() << " " << mp.y() << " "
				<< IPixLeftDownX << " " << IPixLeftDownY ;

    // Check detector limits to correct for pixels outside range.
    int numColumns = topol->ncolumns();  // det module number of cols&rows
    int numRows = topol->nrows();

    IPixRightUpX = numRows > IPixRightUpX ? IPixRightUpX : numRows-1;
    IPixRightUpY = numColumns > IPixRightUpY ? IPixRightUpY : numColumns-1;
    IPixLeftDownX = 0 < IPixLeftDownX ? IPixLeftDownX : 0;
    IPixLeftDownY = 0 < IPixLeftDownY ? IPixLeftDownY : 0;

    x.clear(); // clear temporary integration array
    y.clear();

    // First integrate cahrge strips in x
    int ix; // TT for compatibility
    for (ix = IPixLeftDownX; ix <= IPixRightUpX; ix++) {  // loop over x index
      float xUB, xLB, UpperBound, LowerBound;

      // Why is set to 0 if ix=0, does it meen that we accept charge
      // outside the sensor?
      if (ix == 0 || SigmaX==0.)  // skip for surface segemnts
	LowerBound = 0.;
      else {
	mp = MeasurementPoint( float(ix), 0.0);
	xLB = topol->localPosition(mp).x();
	LowerBound = 1-calcQ((xLB-CloudCenterX)/SigmaX);
      }
       
      if (ix == numRows-1 || SigmaX == 0.)
	UpperBound = 1.;
      else {
	mp = MeasurementPoint(float(ix+1), 0.0);
	xUB = topol->localPosition(mp).x();
	UpperBound = 1. - calcQ((xUB-CloudCenterX)/SigmaX);
      }
      float TotalIntegrationRange = UpperBound - LowerBound; // get strip
      x[ix] = TotalIntegrationRange; // save strip integral
    }

    // Now integarte strips in y
    int iy; // TT for compatibility
    for (iy = IPixLeftDownY; iy <= IPixRightUpY; iy++) { //loope over y ind
      float yUB, yLB, UpperBound, LowerBound;

      if (iy == 0 || SigmaY==0.)
	LowerBound = 0.;
      else {
        mp = MeasurementPoint(0.0, float(iy));
        yLB = topol->localPosition(mp).y();
	LowerBound = 1. - calcQ((yLB-CloudCenterY)/SigmaY);
      }

      if (iy == numColumns-1 || SigmaY==0. )
	UpperBound = 1.;
      else {
        mp = MeasurementPoint(0.0, float(iy+1));
        yUB = topol->localPosition(mp).y();
	UpperBound = 1. - calcQ((yUB-CloudCenterY)/SigmaY);
      }

      float TotalIntegrationRange = UpperBound - LowerBound;
      y[iy] = TotalIntegrationRange; // save strip integral
    }

    // Get the 2D charge integrals by folding x and y strips
    int chan;
    for (ix = IPixLeftDownX; ix <= IPixRightUpX; ix++) {  // loop over x index
      for (iy = IPixLeftDownY; iy <= IPixRightUpY; iy++) { //loope over y ind
        float ChargeFraction = Charge*x[ix]*y[iy];
        if (ChargeFraction > 0.) {
          chan = (pixelFlag) ? PixelDigi::pixelToChannel(ix, iy)
	    : Phase2TrackerDigi::pixelToChannel(ix, iy);  // Get index
          // Load the amplitude
          hit_signal[chan] += ChargeFraction;
	}

	mp = MeasurementPoint(float(ix), float(iy));
	LocalPoint lp = topol->localPosition(mp);
	chan = topol->channel(lp);

	LogDebug("Phase2TrackerDigitizerAlgorithm")
	  << " pixel " << ix << " " << iy << " - "<<" "
	  << chan << " " << ChargeFraction<<" "
	  << mp.x() << " " << mp.y() <<" "
	  << lp.x() << " " << lp.y() << " "  // givex edge position
	  << chan; // edge belongs to previous ?
	ESum += ChargeFraction;  
      }
    }
  }
  // Fill the global map with all hit pixels from this event
  for (auto const & hit_s : hit_signal) {
    int chan =  hit_s.first;
    theSignal[chan] += (makeDigiSimLinks_ ? DigitizerUtility::Amplitude( hit_s.second, &hit, hit_s.second, hitIndex, tofBin) 
                                          : DigitizerUtility::Amplitude( hit_s.second, nullptr, hit_s.second) ) ;
  }
}
// ======================================================================
//
//  Add electronic noise to pixel charge
//
// ======================================================================
void Phase2TrackerDigitizerAlgorithm::add_noise(const Phase2TrackerGeomDetUnit* pixdet,
						float thePixelThreshold) {
  LogDebug("Phase2TrackerDigitizerAlgorithm") << " enter add_noise " << theNoiseInElectrons;
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();
  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();

  // First add Readout noise to hit cells
  for (auto & s : theSignal) {
    float noise  = gaussDistribution_->fire();
    if ((s.second.ampl() + noise) < 0.)
      s.second.set(0);
    else
      s.second += noise;
  }
  
  if (addXtalk) {
    signal_map_type signalNew;
    for (auto const & s : theSignal) {
      float signalInElectrons = s.second.ampl();   // signal in electrons
      std::pair<int,int> hitChan;
      if (pixelFlag) hitChan = PixelDigi::channelToPixel(s.first);
      else hitChan = Phase2TrackerDigi::channelToPixel(s.first);
      
      float signalInElectrons_Xtalk = signalInElectrons * interstripCoupling;     
      
      if (hitChan.first != 0) {	
	std::pair<int,int> XtalkPrev = std::pair<int,int>(hitChan.first-1, hitChan.second);
	int chanXtalkPrev = (pixelFlag) ? PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second)
	  : Phase2TrackerDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
	signalNew.insert(std::pair<int,DigitizerUtility::Amplitude>(chanXtalkPrev, DigitizerUtility::Amplitude(signalInElectrons_Xtalk, nullptr, -1.0)));
      }
      if (hitChan.first < (numRows-1)) {
	std::pair<int,int> XtalkNext = std::pair<int,int>(hitChan.first+1, hitChan.second);
        int chanXtalkNext = (pixelFlag) ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
	  : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
	signalNew.insert(std::pair<int,DigitizerUtility::Amplitude>(chanXtalkNext, DigitizerUtility::Amplitude(signalInElectrons_Xtalk, nullptr, -1.0)));
      }
    }
    for (auto const & l : signalNew) {
      int chan = l.first;
      auto iter = theSignal.find(chan);
      if (iter != theSignal.end()) {
	theSignal[chan] += l.second.ampl();
      }  else {
        theSignal.insert(std::pair<int,DigitizerUtility::Amplitude>(chan, DigitizerUtility::Amplitude(l.second.ampl(), nullptr, -1.0)));
      }
    } 
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
                      otherPixels,
                      rengine_ );

  LogDebug("Phase2TrackerDigitizerAlgorithm")
    <<  " Add noisy pixels " << numRows << " "
    << numColumns << " " << theNoiseInElectrons << " "
    << theThresholdInE_Endcap << "  " << theThresholdInE_Barrel << " " << numberOfPixels << " "
    << otherPixels.size() ;
  
  // Add noisy pixels
  for (mapI = otherPixels.begin(); mapI!= otherPixels.end(); mapI++) {
    int iy = ((*mapI).first) / numRows;
    int ix = ((*mapI).first) - (iy*numRows);
    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) )
      LogWarning("Phase2TrackerDigitizerAlgorithm") << " error in iy " << iy;
    if( ix < 0 || ix > (numRows-1) )
      LogWarning("Phase2TrackerDigitizerAlgorithm") << " error in ix " << ix;

    int chan;
    chan = (pixelFlag) ? PixelDigi::pixelToChannel(ix, iy) : Phase2TrackerDigi::pixelToChannel(ix, iy);

    LogDebug ("Phase2TrackerDigitizerAlgorithm")
      <<" Storing noise = " << (*mapI).first << " " << (*mapI).second
      << " " << ix << " " << iy << " " << chan ;

    if (theSignal[chan] == 0) {
      int noise = int((*mapI).second);
      theSignal[chan] = DigitizerUtility::Amplitude (noise, nullptr, -1.);
    }
  }
}

// ============================================================================
//
// Simulate the readout inefficiencies.
// Delete a selected number of single pixels, dcols and rocs.
void Phase2TrackerDigitizerAlgorithm::pixel_inefficiency(const SubdetEfficiencies& eff,
			                           const Phase2TrackerGeomDetUnit* pixdet,
						   const TrackerTopology *tTopo) {

  uint32_t detID = pixdet->geographicalId().rawId();

  signal_map_type& theSignal = _signal[detID]; // check validity

  // Predefined efficiencies
  float subdetEfficiency  = 1.0;

  // setup the chip indices conversion
  unsigned int Subid=DetId(detID).subdetId();
  if (Subid == PixelSubdetector::PixelBarrel || Subid == StripSubdetector::TOB) { // barrel layers
    unsigned int layerIndex = tTopo->pxbLayer(detID);
    if (layerIndex-1 < eff.barrel_efficiencies.size()) subdetEfficiency = eff.barrel_efficiencies[layerIndex-1];
  } else {                // forward disks
    unsigned int diskIndex = 2 * tTopo->pxfDisk(detID) - tTopo->pxfSide(detID);
    if (diskIndex-1 < eff.endcap_efficiencies.size()) subdetEfficiency  = eff.endcap_efficiencies[diskIndex-1];
  }

  LogDebug ("Phase2TrackerDigitizerAlgorithm") << " enter pixel_inefficiency " << subdetEfficiency;

  // Now loop again over pixels to kill some of them.
  // Loop over hits, amplitude in electrons, channel = coded row,col
  for (auto & s : theSignal) {
    float rand = rengine_->flat();
    if( rand>subdetEfficiency ) {
      // make amplitude =0
      s.second.set(0.); // reset amplitude,
    }
  } 
} 
void Phase2TrackerDigitizerAlgorithm::initializeEvent(CLHEP::HepRandomEngine& eng) {
  if (addNoise || AddPixelInefficiency || fluctuateCharge || addThresholdSmearing) {
    
    gaussDistribution_ = std::make_unique<CLHEP::RandGaussQ>(eng, 0., theReadoutNoise);
  }
  // Threshold smearing with gaussian distribution:
  if (addThresholdSmearing) {
    smearedThreshold_Endcap_ = std::make_unique<CLHEP::RandGaussQ> (eng, theThresholdInE_Endcap , theThresholdSmearing_Endcap);
    smearedThreshold_Barrel_ = std::make_unique<CLHEP::RandGaussQ> (eng, theThresholdInE_Barrel , theThresholdSmearing_Barrel);
  }
  rengine_ = (&eng);
  _signal.clear();
}

// =======================================================================================
//
// Set the drift direction accoring to the Bfield in local det-unit frame
// Works for both barrel and forward pixels.
// Replace the sign convention to fit M.Swartz's formulaes.
// Configurations for barrel and foward pixels possess different tanLorentzAngleperTesla
// parameter value

LocalVector Phase2TrackerDigitizerAlgorithm::DriftDirection(const Phase2TrackerGeomDetUnit* pixdet,
							    const GlobalVector& bfield,
							    const DetId& detId) const {
  Frame detFrame(pixdet->surface().position(),pixdet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);
  float alpha2_Endcap;
  float alpha2_Barrel;
  float alpha2;
  
  float dir_x = 0.0;
  float dir_y = 0.0;
  float dir_z = 0.0;
  float scale = 0.0;

  uint32_t detID = pixdet->geographicalId().rawId();
  unsigned int Sub_detid = DetId(detID).subdetId();

  // Read Lorentz angle from cfg file:
  if (!use_LorentzAngle_DB_) {
    if (alpha2Order) {
      alpha2_Endcap = tanLorentzAnglePerTesla_Endcap*tanLorentzAnglePerTesla_Endcap;
      alpha2_Barrel = tanLorentzAnglePerTesla_Barrel*tanLorentzAnglePerTesla_Barrel;
    } else {
      alpha2_Endcap = 0.0;
      alpha2_Barrel = 0.0;
    }
    
    if (Sub_detid == PixelSubdetector::PixelBarrel || Sub_detid == StripSubdetector::TOB) { // barrel layers
      dir_x = -( tanLorentzAnglePerTesla_Barrel * Bfield.y() + alpha2_Barrel* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_Barrel * Bfield.x() - alpha2_Barrel* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_Barrel* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_Barrel* Bfield.z()*Bfield.z() );

    } else {                                          // forward disks
      dir_x = -( tanLorentzAnglePerTesla_Endcap * Bfield.y() + alpha2_Endcap* Bfield.z()* Bfield.x() );
      dir_y = +( tanLorentzAnglePerTesla_Endcap * Bfield.x() - alpha2_Endcap* Bfield.z()* Bfield.y() );
      dir_z = -(1 + alpha2_Endcap* Bfield.z()*Bfield.z() );
      scale = (1 + alpha2_Endcap* Bfield.z()*Bfield.z() );
    }
  }

  // Read Lorentz angle from DB:
  if (use_LorentzAngle_DB_) {
    float lorentzAngle = SiPixelLorentzAngle_->getLorentzAngle(detId);
    alpha2 = lorentzAngle * lorentzAngle;

    dir_x = -( lorentzAngle * Bfield.y() + alpha2 * Bfield.z()* Bfield.x() );
    dir_y = +( lorentzAngle * Bfield.x() - alpha2 * Bfield.z()* Bfield.y() );
    dir_z = -(1 + alpha2 * Bfield.z()*Bfield.z() );
    scale = (1 + alpha2 * Bfield.z()*Bfield.z() );
  }
  
  LocalVector theDriftDirection = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );
  
  LogDebug ("Phase2TrackerDigitizerAlgorithm") << " The drift direction in local coordinate is "
					       << theDriftDirection ;
  return theDriftDirection;
}

// =============================================================================

void Phase2TrackerDigitizerAlgorithm::pixel_inefficiency_db(uint32_t detID) {
  signal_map_type& theSignal = _signal[detID]; // check validity

  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for (auto  & s : theSignal) {
    std::pair<int,int> ip;
    if (pixelFlag) ip = PixelDigi::channelToPixel(s.first);//get pixel pos
    else ip = Phase2TrackerDigi::channelToPixel(s.first);//get pixel pos
    int row = ip.first;  // X in row
    int col = ip.second; // Y is in col
    //transform to ROC index coordinates
    if (theSiPixelGainCalibrationService_->isDead(detID, col, row)) {
      s.second.set(0.); // reset amplitude,
    }
  }
}

// ==========================================================================

void Phase2TrackerDigitizerAlgorithm::module_killing_conf(uint32_t detID) {
  bool isbad = false;
  int detid = detID;
  std::string Module;
  for ( auto  const & det_m : DeadModules) {
    int Dead_detID = det_m.getParameter<int>("Dead_detID");
    Module = det_m.getParameter<std::string>("Module");
    if (detid == Dead_detID){
      isbad = true;
      break;
    }
  }
  
  if (!isbad) return;

  signal_map_type& theSignal = _signal[detID]; // check validity
  
    
  for (auto & s : theSignal) {
    std::pair<int,int> ip;
    if (pixelFlag) ip  = PixelDigi::channelToPixel(s.first);
    else ip  = Phase2TrackerDigi::channelToPixel(s.first);//get pixel pos

    if (Module == "whole") s.second.set(0.); // reset amplitude
    else if (Module == "tbmA" && ip.first >= 80 && ip.first <= 159) s.second.set(0.);
    else if (Module == "tbmB" && ip.first <= 79) s.second.set(0.);
  }
}
// ==========================================================================
void Phase2TrackerDigitizerAlgorithm::module_killing_DB(uint32_t detID) {
  bool isbad = false;
  
  std::vector<SiPixelQuality::disabledModuleType>disabledModules = SiPixelBadModule_->getBadComponentList();
  
  SiPixelQuality::disabledModuleType badmodule;
  for (size_t id = 0;id < disabledModules.size(); id++) {
    if (detID == disabledModules[id].DetID) {
      isbad = true;
      badmodule = disabledModules[id];
      break;
    }
  }
  
  if (!isbad)
    return;


  signal_map_type& theSignal = _signal[detID]; // check validity
  
  if (badmodule.errorType == 0) { // this is a whole dead module.
    for (auto & s : theSignal) {
      s.second.set(0.); // reset amplitude
    }
  }
  else { // all other module types: half-modules and single ROCs.
    // Get Bad ROC position:
    // follow the example of getBadRocPositions in CondFormats/SiPixelObjects/src/SiPixelQuality.cc
    std::vector<GlobalPixel> badrocpositions (0);
    for (unsigned int j = 0; j < 16; j++) {
      if (SiPixelBadModule_->IsRocBad(detID, j) == true) {
	std::vector<CablingPathToDetUnit> path = map_.product()->pathToDetUnit(detID);
	for (auto const & p : path) {
          const PixelROC* myroc = map_.product()->findItem(p);
          if (myroc->idInDetUnit() == j) {
	    LocalPixel::RocRowCol  local = {39, 25};   //corresponding to center of ROC row, col
	    GlobalPixel global = myroc->toGlobal(LocalPixel(local));
	    badrocpositions.push_back(global);
	    break;
	  }
	}
      }
    }

    
    for (auto & s : theSignal) {
      std::pair<int,int> ip;
      if (pixelFlag) ip  = PixelDigi::channelToPixel(s.first);
      else ip = Phase2TrackerDigi::channelToPixel(s.first);
      
      for (auto const & p : badrocpositions) {
        for (auto & k : badPixels ) {       
	  if ( p.row == k.getParameter<int>("row") && 
	       ip.first == k.getParameter<int>("row")  && 
	       std::abs(ip.second - p.col) < k.getParameter<int>("col")) {s.second.set(0.);}
	}
      }
    }
  }
}

// For premixing
void Phase2TrackerDigitizerAlgorithm::loadAccumulator(unsigned int detId, const std::map<int, float>& accumulator) {
  auto& theSignal = _signal[detId];
  // the input channel is always with PixelDigi definition
  // if needed, that has to be converted to Phase2TrackerDigi convention
  for(const auto& elem: accumulator) {
    auto inserted = theSignal.emplace(elem.first, DigitizerUtility::Amplitude(elem.second, nullptr));
    if(!inserted.second) {
      throw cms::Exception("LogicError") << "Signal was already set for DetId " << detId;
    }
  }
}

void Phase2TrackerDigitizerAlgorithm::digitize(const Phase2TrackerGeomDetUnit* pixdet,
	std::map<int, DigitizerUtility::DigiSimInfo> & digi_map,
	     const TrackerTopology* tTopo) 
{
  uint32_t detID = pixdet->geographicalId().rawId();
  auto it = _signal.find(detID);
  if (it == _signal.end()) return;

  const signal_map_type& theSignal = _signal[detID];


  unsigned int Sub_detid = DetId(detID).subdetId();

  float theThresholdInE = 0.;
  float theHIPThresholdInE = 0.;
  // Define Threshold
  if (Sub_detid == PixelSubdetector::PixelBarrel || Sub_detid == StripSubdetector::TOB) { // Barrel modules
    if (addThresholdSmearing) theThresholdInE = smearedThreshold_Barrel_->fire(); // gaussian smearing
    else theThresholdInE = theThresholdInE_Barrel; // no smearing
    theHIPThresholdInE = theHIPThresholdInE_Barrel;
  } else { // Forward disks modules
    if (addThresholdSmearing) theThresholdInE = smearedThreshold_Endcap_->fire(); // gaussian smearing
    else theThresholdInE = theThresholdInE_Endcap; // no smearing
    theHIPThresholdInE = theHIPThresholdInE_Endcap;
  }

  if (addNoise) add_noise(pixdet, theThresholdInE/theNoiseInElectrons);  // generate noise
  
  // Do only if needed
  if (AddPixelInefficiency && !theSignal.empty()) {
    if (use_ineff_from_db_) 
      pixel_inefficiency_db(detID);
    else                    
      pixel_inefficiency(subdetEfficiencies_, pixdet, tTopo);
  }
  if (use_module_killing_) {
    if (use_deadmodule_DB_)     // remove dead modules using DB
      module_killing_DB(detID);
    else                        // remove dead modules using the list in cfg file
      module_killing_conf(detID);
  }

  // Digitize if the signal is greater than threshold
  for (auto const & s : theSignal) {
    //    DigitizerUtility::Amplitude sig_data = s.second;  
    const DigitizerUtility::Amplitude& sig_data = s.second;
    float signalInElectrons  = sig_data.ampl();
    unsigned short adc;
    if (signalInElectrons >= theThresholdInE) { // check threshold
      adc = convertSignalToAdc(detID, signalInElectrons, theThresholdInE);
      DigitizerUtility::DigiSimInfo info;
      info.sig_tot     = adc;
      info.ot_bit      = ( signalInElectrons  > theHIPThresholdInE ? true : false);
      if (makeDigiSimLinks_ ) {
	for (auto const & l : sig_data.simInfoList()) {
	  float  charge_frac = l.first/signalInElectrons;
          if (l.first > -5.0) info.simInfoList.push_back({charge_frac, l.second.get()});
	}
      }
      digi_map.insert({s.first, info});
    }
  }
}
//
// Scale the Signal using Dual Slope option 
//
int Phase2TrackerDigitizerAlgorithm::convertSignalToAdc(uint32_t detID, float signal_in_elec,float threshold) {
  int signal_in_adc;
  int temp_signal;
  const int max_limit = 10;
  if (thePhase2ReadoutMode == 0) signal_in_adc = theAdcFullScale;
  else { 
    
    if (thePhase2ReadoutMode == -1) {
      temp_signal = std::min( int(signal_in_elec / theElectronPerADC), theAdcFullScale );
    } else {
      // calculate the kink point and the slope
      const int dualslope_param = std::min(abs(thePhase2ReadoutMode), max_limit);
      const int kink_point = int(theAdcFullScale/2) +1;
      temp_signal = std::floor((signal_in_elec-threshold) / theElectronPerADC) + 1;
      if ( temp_signal > kink_point) temp_signal = std::floor((temp_signal - kink_point)/(pow(2, dualslope_param-1))) + kink_point;
    }     
    signal_in_adc = std::min(temp_signal, theAdcFullScale );
    LogInfo("Phase2TrackerDigitizerAlgorithm") << " DetId " << detID
                                               << " signal_in_elec " << signal_in_elec 
					       << " threshold " << threshold << " signal_above_thr " 
					       << (signal_in_elec-threshold) 
					       << " temp conversion " << std::floor((signal_in_elec-threshold)/theElectronPerADC)+1
					       << " signal after slope correction " << temp_signal 
					       << " signal_in_adc " << signal_in_adc;
  } 
  return signal_in_adc; 
}
