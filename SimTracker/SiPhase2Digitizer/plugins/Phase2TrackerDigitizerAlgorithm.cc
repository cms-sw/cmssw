#include <typeinfo>
#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"

#include "CLHEP/Random/RandGaussQ.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

using namespace edm;
using namespace sipixelobjects;

namespace {
  // Mass in MeV
  constexpr double m_pion = 139.571;
  constexpr double m_kaon = 493.677;
  constexpr double m_electron = 0.511;
  constexpr double m_muon = 105.658;
  constexpr double m_proton = 938.272;
}  // namespace

Phase2TrackerDigitizerAlgorithm::Phase2TrackerDigitizerAlgorithm(const edm::ParameterSet& conf_common,
                                                                 const edm::ParameterSet& conf_specific)
    : _signal(),
      makeDigiSimLinks_(conf_common.getUntrackedParameter<bool>("makeDigiSimLinks", true)),
      use_ineff_from_db_(conf_specific.getParameter<bool>("Inefficiency_DB")),
      use_module_killing_(conf_specific.getParameter<bool>("KillModules")),    // boolean to kill or not modules
      use_deadmodule_DB_(conf_specific.getParameter<bool>("DeadModules_DB")),  // boolean to access dead modules from DB
      // boolean to access Lorentz angle from DB
      use_LorentzAngle_DB_(conf_specific.getParameter<bool>("LorentzAngle_DB")),

      // get dead module from cfg file
      deadModules_(use_deadmodule_DB_ ? Parameters() : conf_specific.getParameter<Parameters>("DeadModules")),

      // Common pixel parameters
      // These are parameters which are not likely to be changed
      GeVperElectron_(3.61E-09),                                      // 1 electron(3.61eV, 1keV(277e, mod 9/06 d.k.
      alpha2Order_(conf_specific.getParameter<bool>("Alpha2Order")),  // switch on/off of E.B effect
      addXtalk_(conf_specific.getParameter<bool>("AddXTalk")),
      // Interstrip Coupling - Not used in PixelDigitizerAlgorithm
      interstripCoupling_(conf_specific.getParameter<double>("InterstripCoupling")),

      Sigma0_(conf_specific.getParameter<double>("SigmaZero")),       // Charge diffusion constant 7->3.7
      SigmaCoeff_(conf_specific.getParameter<double>("SigmaCoeff")),  // delta in the diffusion across the strip pitch
      // (set between 0 to 0.9,  0-->flat Sigma0, 1-->Sigma_min=0 & Sigma_max=2*Sigma0
      // D.B.: Dist300 replaced by moduleThickness, may not work with partially depleted sensors but works otherwise
      // Dist300(0.0300),                                          //   normalized to 300micron Silicon

      // Charge integration spread on the collection plane
      clusterWidth_(conf_specific.getParameter<double>("ClusterWidth")),

      // Allowed modes of readout which has following values :
      // 0          ---> Digital or binary readout
      // -1         ---> Analog readout, current digitizer (Inner Pixel) (TDR version) with no threshold subtraction
      // Analog readout with dual slope with the "second" slope being 1/2^(n-1) and threshold subtraction (n = 1, 2, 3,4)
      thePhase2ReadoutMode_(conf_specific.getParameter<int>("Phase2ReadoutMode")),

      // ADC calibration 1adc count(135e.
      // Corresponds to 2adc/kev, 270[e/kev]/135[e/adc](2[adc/kev]
      // Be careful, this parameter is also used in SiPixelDet.cc to
      // calculate the noise in adc counts from noise in electrons.
      // Both defaults should be the same.
      theElectronPerADC_(conf_specific.getParameter<double>("ElectronPerAdc")),

      // ADC saturation value, 255(8bit adc.
      theAdcFullScale_(conf_specific.getParameter<int>("AdcFullScale")),

      // Noise in electrons:
      // Pixel cell noise, relevant for generating noisy pixels
      theNoiseInElectrons_(conf_specific.getParameter<double>("NoiseInElectrons")),

      // Fill readout noise, including all readout chain, relevant for smearing
      theReadoutNoise_(conf_specific.getParameter<double>("ReadoutNoiseInElec")),

      // Threshold in units of noise:
      // thePixelThreshold(conf.getParameter<double>("ThresholdInNoiseUnits")),
      // Pixel threshold in electron units.
      theThresholdInE_Endcap_(conf_specific.getParameter<double>("ThresholdInElectrons_Endcap")),
      theThresholdInE_Barrel_(conf_specific.getParameter<double>("ThresholdInElectrons_Barrel")),

      // Add threshold gaussian smearing:
      theThresholdSmearing_Endcap_(conf_specific.getParameter<double>("ThresholdSmearing_Endcap")),
      theThresholdSmearing_Barrel_(conf_specific.getParameter<double>("ThresholdSmearing_Barrel")),

      // Add HIP Threshold in electron units.
      theHIPThresholdInE_Endcap_(conf_specific.getParameter<double>("HIPThresholdInElectrons_Endcap")),
      theHIPThresholdInE_Barrel_(conf_specific.getParameter<double>("HIPThresholdInElectrons_Barrel")),

      // theTofCut 12.5, cut in particle TOD +/- 12.5ns
      theTofLowerCut_(conf_specific.getParameter<double>("TofLowerCut")),
      theTofUpperCut_(conf_specific.getParameter<double>("TofUpperCut")),

      // Get the Lorentz angle from the cfg file:
      tanLorentzAnglePerTesla_Endcap_(
          use_LorentzAngle_DB_ ? 0.0 : conf_specific.getParameter<double>("TanLorentzAnglePerTesla_Endcap")),
      tanLorentzAnglePerTesla_Barrel_(
          use_LorentzAngle_DB_ ? 0.0 : conf_specific.getParameter<double>("TanLorentzAnglePerTesla_Barrel")),

      // Add noise
      addNoise_(conf_specific.getParameter<bool>("AddNoise")),

      // Add noisy pixels
      addNoisyPixels_(conf_specific.getParameter<bool>("AddNoisyPixels")),

      // Fluctuate charge in track subsegments
      fluctuateCharge_(conf_specific.getUntrackedParameter<bool>("FluctuateCharge", true)),

      // Control the pixel inefficiency
      addPixelInefficiency_(conf_specific.getParameter<bool>("AddInefficiency")),

      // Add threshold gaussian smearing:
      addThresholdSmearing_(conf_specific.getParameter<bool>("AddThresholdSmearing")),

      // Add some pseudo-red damage
      pseudoRadDamage_(conf_specific.exists("PseudoRadDamage") ? conf_specific.getParameter<double>("PseudoRadDamage")
                                                               : double(0.0)),
      pseudoRadDamageRadius_(conf_specific.exists("PseudoRadDamageRadius")
                                 ? conf_specific.getParameter<double>("PseudoRadDamageRadius")
                                 : double(0.0)),

      // delta cutoff in MeV, has to be same as in OSCAR(0.030/cmsim=1.0 MeV
      // tMax(0.030), // In MeV.
      // tMax(conf.getUntrackedParameter<double>("DeltaProductionCut",0.030)),
      tMax_(conf_common.getParameter<double>("DeltaProductionCut")),

      badPixels_(conf_specific.getParameter<Parameters>("CellsToKill")),

      fluctuate_(fluctuateCharge_ ? std::make_unique<SiG4UniversalFluctuation>() : nullptr),
      theNoiser_(addNoise_ ? std::make_unique<GaussianTailNoiseGenerator>() : nullptr),
      theSiPixelGainCalibrationService_(
          use_ineff_from_db_ ? std::make_unique<SiPixelGainCalibrationOfflineSimService>(conf_specific) : nullptr),
      subdetEfficiencies_(conf_specific) {
  LogInfo("Phase2TrackerDigitizerAlgorithm")
      << "Phase2TrackerDigitizerAlgorithm constructed\n"
      << "Configuration parameters:\n"
      << "Threshold/Gain = "
      << "threshold in electron Endcap = " << theThresholdInE_Endcap_
      << "\nthreshold in electron Barrel = " << theThresholdInE_Barrel_ << " ElectronPerADC " << theElectronPerADC_
      << " ADC Scale (in bits) " << theAdcFullScale_ << " The delta cut-off is set to " << tMax_ << " pix-inefficiency "
      << addPixelInefficiency_;
}

Phase2TrackerDigitizerAlgorithm::~Phase2TrackerDigitizerAlgorithm() {
  LogDebug("Phase2TrackerDigitizerAlgorithm") << "Phase2TrackerDigitizerAlgorithm deleted";
}

Phase2TrackerDigitizerAlgorithm::SubdetEfficiencies::SubdetEfficiencies(const edm::ParameterSet& conf) {
  barrel_efficiencies = conf.getParameter<std::vector<double> >("EfficiencyFactors_Barrel");
  endcap_efficiencies = conf.getParameter<std::vector<double> >("EfficiencyFactors_Endcap");
}
// =================================================================
//
// Generate primary ionization along the track segment.
// Divide the track into small sub-segments
//
// =================================================================
void Phase2TrackerDigitizerAlgorithm::primary_ionization(
    const PSimHit& hit, std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points) const {
  // Straight line approximation for trajectory inside active media
  constexpr float SegmentLength = 0.0010;  // in cm (10 microns)
  // Get the 3D segment direction vector
  LocalVector direction = hit.exitPoint() - hit.entryPoint();

  float eLoss = hit.energyLoss();  // Eloss in GeV
  float length = direction.mag();  // Track length in Silicon

  int NumberOfSegments = static_cast<int>(length / SegmentLength);  // Number of segments
  if (NumberOfSegments < 1)
    NumberOfSegments = 1;
  LogDebug("Phase2TrackerDigitizerAlgorithm")
      << "enter primary_ionzation " << NumberOfSegments << " shift = " << hit.exitPoint().x() - hit.entryPoint().x()
      << " " << hit.exitPoint().y() - hit.entryPoint().y() << " " << hit.exitPoint().z() - hit.entryPoint().z() << " "
      << hit.particleType() << " " << hit.pabs();

  std::vector<float> elossVector;  // Eloss vector
  elossVector.reserve(NumberOfSegments);
  if (fluctuateCharge_) {
    // Generate fluctuated charge points
    fluctuateEloss(hit.particleType(), hit.pabs(), eLoss, length, NumberOfSegments, elossVector);
  }
  ionization_points.reserve(NumberOfSegments);  // set size

  // loop over segments
  for (size_t i = 0; i < elossVector.size(); ++i) {
    // Divide the segment into equal length subsegments
    Local3DPoint point = hit.entryPoint() + ((i + 0.5) / NumberOfSegments) * direction;
    float energy = fluctuateCharge_ ? elossVector[i] / GeVperElectron_  // Convert charge to elec.
                                    : eLoss / GeVperElectron_ / NumberOfSegments;

    DigitizerUtility::EnergyDepositUnit edu(energy, point);  // define position,energy point
    ionization_points.push_back(edu);                        // save
    LogDebug("Phase2TrackerDigitizerAlgorithm")
        << i << " " << edu.x() << " " << edu.y() << " " << edu.z() << " " << edu.energy();
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
                                                     std::vector<float>& elossVector) const {
  // Get dedx for this track
  //float dedx;
  //if( length > 0.) dedx = eloss/length;
  //else dedx = eloss;

  double particleMass = ::m_pion;  // Mass in MeV, assume pion
  pid = std::abs(pid);
  if (pid != 211) {  // Mass in MeV
    if (pid == 11)
      particleMass = ::m_electron;
    else if (pid == 13)
      particleMass = ::m_muon;
    else if (pid == 321)
      particleMass = ::m_kaon;
    else if (pid == 2212)
      particleMass = ::m_proton;
  }
  // What is the track segment length.
  float segmentLength = length / NumberOfSegs;

  // Generate charge fluctuations.
  float sum = 0.;
  double segmentEloss = (1000. * eloss) / NumberOfSegs;  //eloss in MeV
  for (int i = 0; i < NumberOfSegs; ++i) {
    //       material,*,   momentum,energy,*, *,  mass
    //myglandz_(14.,segmentLength,2.,2.,dedx,de,0.14);
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    double deltaCutoff = tMax_;  // the cutoff is sometimes redefined inside, so fix it.
    float de = fluctuate_->SampleFluctuations(particleMomentum * 1000.,
                                              particleMass,
                                              deltaCutoff,
                                              segmentLength * 10.,
                                              segmentEloss,
                                              rengine_) /
               1000.;  //convert to GeV
    elossVector.push_back(de);
    sum += de;
  }
  if (sum > 0.) {  // if fluctuations give eloss>0.
    // Rescale to the same total eloss
    float ratio = eloss / sum;
    std::transform(
        std::begin(elossVector), std::end(elossVector), std::begin(elossVector), [&ratio](auto const& c) -> float {
          return c * ratio;
        });  // use a simple lambda expression
  } else {   // if fluctuations gives 0 eloss
    float averageEloss = eloss / NumberOfSegs;
    std::fill(std::begin(elossVector), std::end(elossVector), averageEloss);
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

  collection_points.reserve(ionization_points.size());                     // set size
  LocalVector driftDir = DriftDirection(pixdet, bfield, hit.detUnitId());  // get the charge drift direction
  if (driftDir.z() == 0.) {
    LogWarning("Phase2TrackerDigitizerAlgorithm") << " pxlx: drift in z is zero ";
    return;
  }

  float TanLorenzAngleX = driftDir.x();                                       // tangent of Lorentz angle
  float TanLorenzAngleY = 0.;                                                 // force to 0, driftDir.y()/driftDir.z();
  float dir_z = driftDir.z();                                                 // The z drift direction
  float CosLorenzAngleX = 1. / std::sqrt(1. + std::pow(TanLorenzAngleX, 2));  // cosine to estimate the path length
  float CosLorenzAngleY = 1.;
  if (alpha2Order_) {
    TanLorenzAngleY = driftDir.y();
    CosLorenzAngleY = 1. / std::sqrt(1. + std::pow(TanLorenzAngleY, 2));  // cosine
  }

  float moduleThickness = pixdet->specificSurface().bounds().thickness();
  float stripPitch = pixdet->specificTopology().pitch().first;

  LogDebug("Phase2TrackerDigitizerAlgorithm")
      << " Lorentz Tan " << TanLorenzAngleX << " " << TanLorenzAngleY << " " << CosLorenzAngleX << " "
      << CosLorenzAngleY << " " << moduleThickness * TanLorenzAngleX << " " << driftDir;

  for (auto const& val : ionization_points) {
    // position
    float SegX = val.x(), SegY = val.y(), SegZ = val.z();

    // Distance from the collection plane
    // DriftDistance = (moduleThickness/2. + SegZ); // Drift to -z
    // Include explixitely the E drift direction (for CMS dir_z=-1)

    // Distance between charge generation and collection
    float driftDistance = moduleThickness / 2. - (dir_z * SegZ);  // Drift to -z

    if (driftDistance < 0.)
      driftDistance = 0.;
    else if (driftDistance > moduleThickness)
      driftDistance = moduleThickness;

    // Assume full depletion now, partial depletion will come later.
    float XDriftDueToMagField = driftDistance * TanLorenzAngleX;
    float YDriftDueToMagField = driftDistance * TanLorenzAngleY;

    // Shift cloud center
    float CloudCenterX = SegX + XDriftDueToMagField;
    float CloudCenterY = SegY + YDriftDueToMagField;

    // Calculate how long is the charge drift path
    // Actual Drift Lentgh
    float driftLength =
        std::sqrt(std::pow(driftDistance, 2) + std::pow(XDriftDueToMagField, 2) + std::pow(YDriftDueToMagField, 2));

    // What is the charge diffusion after this path
    // Sigma0=0.00037 is for 300um thickness (make sure moduleThickness is in [cm])
    float Sigma = std::sqrt(driftLength / moduleThickness) * Sigma0_ * moduleThickness / 0.0300;
    // D.B.: sigmaCoeff=0 means no modulation
    if (SigmaCoeff_)
      Sigma *= (SigmaCoeff_ * std::pow(cos(SegX * M_PI / stripPitch), 2) + 1);
    // NB: divided by 4 to get a periodicity of stripPitch

    // Project the diffusion sigma on the collection plane
    float Sigma_x = Sigma / CosLorenzAngleX;
    float Sigma_y = Sigma / CosLorenzAngleY;

    // Insert a charge loss due to Rad Damage here
    float energyOnCollector = val.energy();  // The energy that reaches the collector

    // pseudoRadDamage
    if (pseudoRadDamage_) {
      float moduleRadius = pixdet->surface().position().perp();
      if (moduleRadius <= pseudoRadDamageRadius_) {
        float kValue = pseudoRadDamage_ / std::pow(moduleRadius, 2);
        energyOnCollector *= exp(-1 * kValue * driftDistance / moduleThickness);
      }
    }
    LogDebug("Phase2TrackerDigitizerAlgorithm")
        << "Dift DistanceZ = " << driftDistance << " module thickness = " << moduleThickness
        << " Start Energy = " << val.energy() << " Energy after loss= " << energyOnCollector;
    DigitizerUtility::SignalPoint sp(CloudCenterX, CloudCenterY, Sigma_x, Sigma_y, hit.tof(), energyOnCollector);

    // Load the Charge distribution parameters
    collection_points.push_back(sp);
  }
}

// ====================================================================
//
// Induce the signal on the collection plane of the active sensor area.
void Phase2TrackerDigitizerAlgorithm::induce_signal(
    const PSimHit& hit,
    const size_t hitIndex,
    const uint32_t tofBin,
    const Phase2TrackerGeomDetUnit* pixdet,
    const std::vector<DigitizerUtility::SignalPoint>& collection_points) {
  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];

  LogDebug("Phase2TrackerDigitizerAlgorithm")
      << " enter induce_signal, " << topol->pitch().first << " " << topol->pitch().second;

  // local map to store pixels hit by 1 Hit.
  using hit_map_type = std::map<int, float, std::less<int> >;
  hit_map_type hit_signal;

  // Assign signals to readout channels and store sorted by channel number
  // Iterate over collection points on the collection plane
  for (auto const& v : collection_points) {
    float CloudCenterX = v.position().x();  // Charge position in x
    float CloudCenterY = v.position().y();  //                 in y
    float SigmaX = v.sigma_x();             // Charge spread in x
    float SigmaY = v.sigma_y();             //               in y
    float Charge = v.amplitude();           // Charge amplitude

    LogDebug("Phase2TrackerDigitizerAlgorithm") << " cloud " << v.position().x() << " " << v.position().y() << " "
                                                << v.sigma_x() << " " << v.sigma_y() << " " << v.amplitude();

    // Find the maximum cloud spread in 2D plane , assume 3*sigma
    float CloudRight = CloudCenterX + clusterWidth_ * SigmaX;
    float CloudLeft = CloudCenterX - clusterWidth_ * SigmaX;
    float CloudUp = CloudCenterY + clusterWidth_ * SigmaY;
    float CloudDown = CloudCenterY - clusterWidth_ * SigmaY;

    // Define 2D cloud limit points
    LocalPoint PointRightUp = LocalPoint(CloudRight, CloudUp);
    LocalPoint PointLeftDown = LocalPoint(CloudLeft, CloudDown);

    // This points can be located outside the sensor area.
    // The conversion to measurement point does not check for that
    // so the returned pixel index might be wrong (outside range).
    // We rely on the limits check below to fix this.
    // But remember whatever we do here THE CHARGE OUTSIDE THE ACTIVE
    // PIXEL ARE IS LOST, it should not be collected.

    // Convert the 2D points to pixel indices
    MeasurementPoint mp = topol->measurementPosition(PointRightUp);
    int IPixRightUpX = static_cast<int>(std::floor(mp.x()));  // cast reqd.
    int IPixRightUpY = static_cast<int>(std::floor(mp.y()));
    LogDebug("Phase2TrackerDigitizerAlgorithm")
        << " right-up " << PointRightUp << " " << mp.x() << " " << mp.y() << " " << IPixRightUpX << " " << IPixRightUpY;

    mp = topol->measurementPosition(PointLeftDown);
    int IPixLeftDownX = static_cast<int>(std::floor(mp.x()));
    int IPixLeftDownY = static_cast<int>(std::floor(mp.y()));
    LogDebug("Phase2TrackerDigitizerAlgorithm") << " left-down " << PointLeftDown << " " << mp.x() << " " << mp.y()
                                                << " " << IPixLeftDownX << " " << IPixLeftDownY;

    // Check detector limits to correct for pixels outside range.
    int numColumns = topol->ncolumns();  // det module number of cols&rows
    int numRows = topol->nrows();

    IPixRightUpX = numRows > IPixRightUpX ? IPixRightUpX : numRows - 1;
    IPixRightUpY = numColumns > IPixRightUpY ? IPixRightUpY : numColumns - 1;
    IPixLeftDownX = 0 < IPixLeftDownX ? IPixLeftDownX : 0;
    IPixLeftDownY = 0 < IPixLeftDownY ? IPixLeftDownY : 0;

    // First integrate charge strips in x
    hit_map_type x;
    for (int ix = IPixLeftDownX; ix <= IPixRightUpX; ++ix) {  // loop over x index
      float xLB, LowerBound;
      // Why is set to 0 if ix=0, does it meen that we accept charge
      // outside the sensor?
      if (ix == 0 || SigmaX == 0.) {  // skip for surface segemnts
        LowerBound = 0.;
      } else {
        mp = MeasurementPoint(ix, 0.0);
        xLB = topol->localPosition(mp).x();
        LowerBound = 1 - calcQ((xLB - CloudCenterX) / SigmaX);
      }

      float xUB, UpperBound;
      if (ix == numRows - 1 || SigmaX == 0.) {
        UpperBound = 1.;
      } else {
        mp = MeasurementPoint(ix + 1, 0.0);
        xUB = topol->localPosition(mp).x();
        UpperBound = 1. - calcQ((xUB - CloudCenterX) / SigmaX);
      }
      float TotalIntegrationRange = UpperBound - LowerBound;  // get strip
      x.emplace(ix, TotalIntegrationRange);                   // save strip integral
    }

    // Now integrate strips in y
    hit_map_type y;
    for (int iy = IPixLeftDownY; iy <= IPixRightUpY; ++iy) {  // loop over y index
      float yLB, LowerBound;
      if (iy == 0 || SigmaY == 0.) {
        LowerBound = 0.;
      } else {
        mp = MeasurementPoint(0.0, iy);
        yLB = topol->localPosition(mp).y();
        LowerBound = 1. - calcQ((yLB - CloudCenterY) / SigmaY);
      }

      float yUB, UpperBound;
      if (iy == numColumns - 1 || SigmaY == 0.) {
        UpperBound = 1.;
      } else {
        mp = MeasurementPoint(0.0, iy + 1);
        yUB = topol->localPosition(mp).y();
        UpperBound = 1. - calcQ((yUB - CloudCenterY) / SigmaY);
      }

      float TotalIntegrationRange = UpperBound - LowerBound;
      y.emplace(iy, TotalIntegrationRange);  // save strip integral
    }

    // Get the 2D charge integrals by folding x and y strips
    for (int ix = IPixLeftDownX; ix <= IPixRightUpX; ++ix) {    // loop over x index
      for (int iy = IPixLeftDownY; iy <= IPixRightUpY; ++iy) {  // loop over y index
        float ChargeFraction = Charge * x[ix] * y[iy];
        int chanFired = -1;
        if (ChargeFraction > 0.) {
          chanFired =
              pixelFlag_ ? PixelDigi::pixelToChannel(ix, iy) : Phase2TrackerDigi::pixelToChannel(ix, iy);  // Get index
          // Load the amplitude
          hit_signal[chanFired] += ChargeFraction;
        }

        mp = MeasurementPoint(ix, iy);
        LocalPoint lp = topol->localPosition(mp);
        int chan = topol->channel(lp);

        LogDebug("Phase2TrackerDigitizerAlgorithm")
            << " pixel " << ix << " " << iy << " - "
            << " " << chanFired << " " << ChargeFraction << " " << mp.x() << " " << mp.y() << " " << lp.x() << " "
            << lp.y() << " "  // givex edge position
            << chan;          // edge belongs to previous ?
      }
    }
  }
  // Fill the global map with all hit pixels from this event
  for (auto const& hit_s : hit_signal) {
    int chan = hit_s.first;
    theSignal[chan] +=
        (makeDigiSimLinks_ ? DigitizerUtility::Amplitude(hit_s.second, &hit, hit_s.second, hitIndex, tofBin)
                           : DigitizerUtility::Amplitude(hit_s.second, nullptr, hit_s.second));
  }
}
// ======================================================================
//
//  Add electronic noise to pixel charge
//
// ======================================================================
void Phase2TrackerDigitizerAlgorithm::add_noise(const Phase2TrackerGeomDetUnit* pixdet) {
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  for (auto& s : theSignal) {
    float noise = gaussDistribution_->fire();
    if ((s.second.ampl() + noise) < 0.)
      s.second.set(0);
    else
      s.second += noise;
  }
}
// ======================================================================
//
//  Add  Cross-talk contribution
//
// ======================================================================
void Phase2TrackerDigitizerAlgorithm::add_cross_talk(const Phase2TrackerGeomDetUnit* pixdet) {
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  signal_map_type signalNew;
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();
  int numRows = topol->nrows();

  for (auto& s : theSignal) {
    float signalInElectrons = s.second.ampl();  // signal in electrons

    std::pair<int, int> hitChan;
    if (pixelFlag_)
      hitChan = PixelDigi::channelToPixel(s.first);
    else
      hitChan = Phase2TrackerDigi::channelToPixel(s.first);

    float signalInElectrons_Xtalk = signalInElectrons * interstripCoupling_;
    // subtract the charge which will be shared
    s.second.set(signalInElectrons - signalInElectrons_Xtalk);

    if (hitChan.first != 0) {
      auto XtalkPrev = std::make_pair(hitChan.first - 1, hitChan.second);
      int chanXtalkPrev = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
      signalNew.emplace(chanXtalkPrev, DigitizerUtility::Amplitude(signalInElectrons_Xtalk, nullptr, -1.0));
    }
    if (hitChan.first < numRows - 1) {
      auto XtalkNext = std::make_pair(hitChan.first + 1, hitChan.second);
      int chanXtalkNext = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
      signalNew.emplace(chanXtalkNext, DigitizerUtility::Amplitude(signalInElectrons_Xtalk, nullptr, -1.0));
    }
  }
  for (auto const& l : signalNew) {
    int chan = l.first;
    auto iter = theSignal.find(chan);
    if (iter != theSignal.end()) {
      theSignal[chan] += l.second.ampl();
    } else {
      theSignal.emplace(chan, DigitizerUtility::Amplitude(l.second.ampl(), nullptr, -1.0));
    }
  }
}

// ======================================================================
//
//  Add noise on non-hit cells
//
// ======================================================================
void Phase2TrackerDigitizerAlgorithm::add_noisy_cells(const Phase2TrackerGeomDetUnit* pixdet, float thePixelThreshold) {
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();

  int numColumns = topol->ncolumns();  // det module number of cols&rows
  int numRows = topol->nrows();

  int numberOfPixels = numRows * numColumns;
  std::map<int, float, std::less<int> > otherPixels;

  theNoiser_->generate(numberOfPixels,
                       thePixelThreshold,     //thr. in un. of nois
                       theNoiseInElectrons_,  // noise in elec.
                       otherPixels,
                       rengine_);

  LogDebug("Phase2TrackerDigitizerAlgorithm")
      << " Add noisy pixels " << numRows << " " << numColumns << " " << theNoiseInElectrons_ << " "
      << theThresholdInE_Endcap_ << "  " << theThresholdInE_Barrel_ << " " << numberOfPixels << " "
      << otherPixels.size();

  // Add noisy pixels
  for (auto const& el : otherPixels) {
    int iy = el.first / numRows;
    if (iy < 0 || iy > numColumns - 1)
      LogWarning("Phase2TrackerDigitizerAlgorithm") << " error in iy " << iy;

    int ix = el.first - iy * numRows;
    if (ix < 0 || ix > numRows - 1)
      LogWarning("Phase2TrackerDigitizerAlgorithm") << " error in ix " << ix;

    int chan = pixelFlag_ ? PixelDigi::pixelToChannel(ix, iy) : Phase2TrackerDigi::pixelToChannel(ix, iy);

    LogDebug("Phase2TrackerDigitizerAlgorithm")
        << " Storing noise = " << el.first << " " << el.second << " " << ix << " " << iy << " " << chan;

    if (theSignal[chan] == 0)
      theSignal[chan] = DigitizerUtility::Amplitude(el.second, nullptr, -1.);
  }
}
// ============================================================================
//
// Simulate the readout inefficiencies.
// Delete a selected number of single pixels, dcols and rocs.
void Phase2TrackerDigitizerAlgorithm::pixel_inefficiency(const SubdetEfficiencies& eff,
                                                         const Phase2TrackerGeomDetUnit* pixdet,
                                                         const TrackerTopology* tTopo) {
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];  // check validity

  // Predefined efficiencies
  float subdetEfficiency = 1.0;

  // setup the chip indices conversion
  uint32_t Subid = DetId(detID).subdetId();
  if (Subid == PixelSubdetector::PixelBarrel || Subid == StripSubdetector::TOB) {  // barrel layers
    uint32_t layerIndex = tTopo->pxbLayer(detID);
    if (layerIndex - 1 < eff.barrel_efficiencies.size())
      subdetEfficiency = eff.barrel_efficiencies[layerIndex - 1];
  } else {  // forward disks
    uint32_t diskIndex = 2 * tTopo->pxfDisk(detID) - tTopo->pxfSide(detID);
    if (diskIndex - 1 < eff.endcap_efficiencies.size())
      subdetEfficiency = eff.endcap_efficiencies[diskIndex - 1];
  }

  LogDebug("Phase2TrackerDigitizerAlgorithm") << " enter pixel_inefficiency " << subdetEfficiency;

  // Now loop again over pixels to kill some of them.
  // Loop over hits, amplitude in electrons, channel = coded row,col
  for (auto& s : theSignal) {
    float rand = rengine_->flat();
    if (rand > subdetEfficiency) {
      // make amplitude =0
      s.second.set(0.);  // reset amplitude
    }
  }
}
void Phase2TrackerDigitizerAlgorithm::initializeEvent(CLHEP::HepRandomEngine& eng) {
  if (addNoise_ || addPixelInefficiency_ || fluctuateCharge_ || addThresholdSmearing_) {
    gaussDistribution_ = std::make_unique<CLHEP::RandGaussQ>(eng, 0., theReadoutNoise_);
  }
  // Threshold smearing with gaussian distribution:
  if (addThresholdSmearing_) {
    smearedThreshold_Endcap_ =
        std::make_unique<CLHEP::RandGaussQ>(eng, theThresholdInE_Endcap_, theThresholdSmearing_Endcap_);
    smearedThreshold_Barrel_ =
        std::make_unique<CLHEP::RandGaussQ>(eng, theThresholdInE_Barrel_, theThresholdSmearing_Barrel_);
  }
  rengine_ = &eng;
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
  Frame detFrame(pixdet->surface().position(), pixdet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);

  float dir_x = 0.0;
  float dir_y = 0.0;
  float dir_z = 0.0;
  float scale = 1.0;

  uint32_t detID = pixdet->geographicalId().rawId();
  uint32_t Sub_detid = DetId(detID).subdetId();

  // Read Lorentz angle from DB:
  if (use_LorentzAngle_DB_) {
    float lorentzAngle = SiPixelLorentzAngle_->getLorentzAngle(detId);
    float alpha2 = std::pow(lorentzAngle, 2);

    dir_x = -(lorentzAngle * Bfield.y() + alpha2 * Bfield.z() * Bfield.x());
    dir_y = +(lorentzAngle * Bfield.x() - alpha2 * Bfield.z() * Bfield.y());
    dir_z = -(1 + alpha2 * std::pow(Bfield.z(), 2));
    scale = (1 + alpha2 * std::pow(Bfield.z(), 2));
  } else {
    // Read Lorentz angle from cfg file:
    float alpha2_Endcap = 0.0;
    float alpha2_Barrel = 0.0;
    if (alpha2Order_) {
      alpha2_Endcap = std::pow(tanLorentzAnglePerTesla_Endcap_, 2);
      alpha2_Barrel = std::pow(tanLorentzAnglePerTesla_Barrel_, 2);
    }

    if (Sub_detid == PixelSubdetector::PixelBarrel || Sub_detid == StripSubdetector::TOB) {  // barrel layers
      dir_x = -(tanLorentzAnglePerTesla_Barrel_ * Bfield.y() + alpha2_Barrel * Bfield.z() * Bfield.x());
      dir_y = +(tanLorentzAnglePerTesla_Barrel_ * Bfield.x() - alpha2_Barrel * Bfield.z() * Bfield.y());
      dir_z = -(1 + alpha2_Barrel * std::pow(Bfield.z(), 2));
      scale = (1 + alpha2_Barrel * std::pow(Bfield.z(), 2));

    } else {  // forward disks
      dir_x = -(tanLorentzAnglePerTesla_Endcap_ * Bfield.y() + alpha2_Endcap * Bfield.z() * Bfield.x());
      dir_y = +(tanLorentzAnglePerTesla_Endcap_ * Bfield.x() - alpha2_Endcap * Bfield.z() * Bfield.y());
      dir_z = -(1 + alpha2_Endcap * std::pow(Bfield.z(), 2));
      scale = (1 + alpha2_Endcap * std::pow(Bfield.z(), 2));
    }
  }

  LocalVector theDriftDirection = LocalVector(dir_x / scale, dir_y / scale, dir_z / scale);
  LogDebug("Phase2TrackerDigitizerAlgorithm") << " The drift direction in local coordinate is " << theDriftDirection;
  return theDriftDirection;
}

// =============================================================================

void Phase2TrackerDigitizerAlgorithm::pixel_inefficiency_db(uint32_t detID) {
  signal_map_type& theSignal = _signal[detID];  // check validity

  // Loop over hit pixels, amplitude in electrons, channel = coded row,col
  for (auto& s : theSignal) {
    std::pair<int, int> ip;
    if (pixelFlag_)
      ip = PixelDigi::channelToPixel(s.first);  //get pixel pos
    else
      ip = Phase2TrackerDigi::channelToPixel(s.first);  //get pixel pos

    int row = ip.first;   // X in row
    int col = ip.second;  // Y is in col
    // transform to ROC index coordinates
    if (theSiPixelGainCalibrationService_->isDead(detID, col, row))
      s.second.set(0.);  // reset amplitude
  }
}

// ==========================================================================

void Phase2TrackerDigitizerAlgorithm::module_killing_conf(uint32_t detID) {
  bool isbad = false;
  int detid = detID;
  std::string Module;
  for (auto const& det_m : deadModules_) {
    int Dead_detID = det_m.getParameter<int>("Dead_detID");
    Module = det_m.getParameter<std::string>("Module");
    if (detid == Dead_detID) {
      isbad = true;
      break;
    }
  }

  if (!isbad)
    return;

  signal_map_type& theSignal = _signal[detID];  // check validity
  for (auto& s : theSignal) {
    std::pair<int, int> ip;
    if (pixelFlag_)
      ip = PixelDigi::channelToPixel(s.first);
    else
      ip = Phase2TrackerDigi::channelToPixel(s.first);  //get pixel pos

    if (Module == "whole" || (Module == "tbmA" && ip.first >= 80 && ip.first <= 159) ||
        (Module == "tbmB" && ip.first <= 79))
      s.second.set(0.);
  }
}
// ==========================================================================
void Phase2TrackerDigitizerAlgorithm::module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) {
  bool isbad = false;
  uint32_t detID = pixdet->geographicalId().rawId();
  int ncol = pixdet->specificTopology().ncolumns();
  if (ncol < 0)
    return;
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = SiPixelBadModule_->getBadComponentList();

  SiPixelQuality::disabledModuleType badmodule;
  for (size_t id = 0; id < disabledModules.size(); id++) {
    if (detID == disabledModules[id].DetID) {
      isbad = true;
      badmodule = disabledModules[id];
      break;
    }
  }

  if (!isbad)
    return;

  signal_map_type& theSignal = _signal[detID];  // check validity
  if (badmodule.errorType == 0) {               // this is a whole dead module.
    for (auto& s : theSignal)
      s.second.set(0.);  // reset amplitude
  } else {               // all other module types: half-modules and single ROCs.
    // Get Bad ROC position:
    // follow the example of getBadRocPositions in CondFormats/SiPixelObjects/src/SiPixelQuality.cc
    std::vector<GlobalPixel> badrocpositions;
    for (size_t j = 0; j < static_cast<size_t>(ncol); j++) {
      if (SiPixelBadModule_->IsRocBad(detID, j)) {
        std::vector<CablingPathToDetUnit> path = fedCablingMap_.product()->pathToDetUnit(detID);
        for (auto const& p : path) {
          const PixelROC* myroc = fedCablingMap_.product()->findItem(p);
          if (myroc->idInDetUnit() == j) {
            LocalPixel::RocRowCol local = {39, 25};  //corresponding to center of ROC row, col
            GlobalPixel global = myroc->toGlobal(LocalPixel(local));
            badrocpositions.push_back(global);
            break;
          }
        }
      }
    }

    for (auto& s : theSignal) {
      std::pair<int, int> ip;
      if (pixelFlag_)
        ip = PixelDigi::channelToPixel(s.first);
      else
        ip = Phase2TrackerDigi::channelToPixel(s.first);

      for (auto const& p : badrocpositions) {
        for (auto& k : badPixels_) {
          if (p.row == k.getParameter<int>("row") && ip.first == k.getParameter<int>("row") &&
              std::abs(ip.second - p.col) < k.getParameter<int>("col")) {
            s.second.set(0.);
          }
        }
      }
    }
  }
}

// For premixing
void Phase2TrackerDigitizerAlgorithm::loadAccumulator(uint32_t detId, const std::map<int, float>& accumulator) {
  auto& theSignal = _signal[detId];
  // the input channel is always with PixelDigi definition
  // if needed, that has to be converted to Phase2TrackerDigi convention
  for (const auto& elem : accumulator) {
    auto inserted = theSignal.emplace(elem.first, DigitizerUtility::Amplitude(elem.second, nullptr));
    if (!inserted.second) {
      throw cms::Exception("LogicError") << "Signal was already set for DetId " << detId;
    }
  }
}

void Phase2TrackerDigitizerAlgorithm::digitize(const Phase2TrackerGeomDetUnit* pixdet,
                                               std::map<int, DigitizerUtility::DigiSimInfo>& digi_map,
                                               const TrackerTopology* tTopo) {
  uint32_t detID = pixdet->geographicalId().rawId();
  auto it = _signal.find(detID);
  if (it == _signal.end())
    return;

  const signal_map_type& theSignal = _signal[detID];

  uint32_t Sub_detid = DetId(detID).subdetId();

  float theThresholdInE = 0.;
  float theHIPThresholdInE = 0.;
  // Define Threshold
  if (Sub_detid == PixelSubdetector::PixelBarrel || Sub_detid == StripSubdetector::TOB) {  // Barrel modules
    theThresholdInE = addThresholdSmearing_ ? smearedThreshold_Barrel_->fire()             // gaussian smearing
                                            : theThresholdInE_Barrel_;                     // no smearing
    theHIPThresholdInE = theHIPThresholdInE_Barrel_;
  } else {                                                                      // Forward disks modules
    theThresholdInE = addThresholdSmearing_ ? smearedThreshold_Endcap_->fire()  // gaussian smearing
                                            : theThresholdInE_Endcap_;          // no smearing
    theHIPThresholdInE = theHIPThresholdInE_Endcap_;
  }

  //  if (addNoise) add_noise(pixdet, theThresholdInE/theNoiseInElectrons_);  // generate noise
  if (addNoise_)
    add_noise(pixdet);  // generate noise
  if (addXtalk_)
    add_cross_talk(pixdet);
  if (addNoisyPixels_)
    add_noisy_cells(pixdet, theHIPThresholdInE / theElectronPerADC_);

  // Do only if needed
  if (addPixelInefficiency_ && !theSignal.empty()) {
    if (use_ineff_from_db_)
      pixel_inefficiency_db(detID);
    else
      pixel_inefficiency(subdetEfficiencies_, pixdet, tTopo);
  }
  if (use_module_killing_) {
    if (use_deadmodule_DB_)  // remove dead modules using DB
      module_killing_DB(pixdet);
    else  // remove dead modules using the list in cfg file
      module_killing_conf(detID);
  }

  // Digitize if the signal is greater than threshold
  for (auto const& s : theSignal) {
    const DigitizerUtility::Amplitude& sig_data = s.second;
    float signalInElectrons = sig_data.ampl();
    if (signalInElectrons >= theThresholdInE) {  // check threshold
      DigitizerUtility::DigiSimInfo info;
      info.sig_tot = convertSignalToAdc(detID, signalInElectrons, theThresholdInE);  // adc
      info.ot_bit = signalInElectrons > theHIPThresholdInE ? true : false;
      if (makeDigiSimLinks_) {
        for (auto const& l : sig_data.simInfoList()) {
          float charge_frac = l.first / signalInElectrons;
          if (l.first > -5.0)
            info.simInfoList.push_back({charge_frac, l.second.get()});
        }
      }
      digi_map.insert({s.first, info});
    }
  }
}
//
// Scale the Signal using Dual Slope option
//
int Phase2TrackerDigitizerAlgorithm::convertSignalToAdc(uint32_t detID, float signal_in_elec, float threshold) {
  int signal_in_adc;
  int temp_signal;
  const int max_limit = 10;
  if (thePhase2ReadoutMode_ == 0) {
    signal_in_adc = theAdcFullScale_;
  } else {
    if (thePhase2ReadoutMode_ == -1) {
      temp_signal = std::min(static_cast<int>(signal_in_elec / theElectronPerADC_), theAdcFullScale_);
    } else {
      // calculate the kink point and the slope
      int dualslope_param = std::min(std::abs(thePhase2ReadoutMode_), max_limit);
      int kink_point = static_cast<int>(theAdcFullScale_ / 2) + 1;
      temp_signal = std::floor((signal_in_elec - threshold) / theElectronPerADC_) + 1;
      if (temp_signal > kink_point)
        temp_signal = std::floor((temp_signal - kink_point) / (pow(2, dualslope_param - 1))) + kink_point;
    }
    signal_in_adc = std::min(temp_signal, theAdcFullScale_);
    LogInfo("Phase2TrackerDigitizerAlgorithm")
        << " DetId " << detID << " signal_in_elec " << signal_in_elec << " threshold " << threshold
        << " signal_above_thr " << signal_in_elec - threshold << " temp conversion "
        << std::floor((signal_in_elec - threshold) / theElectronPerADC_) + 1 << " signal after slope correction "
        << temp_signal << " signal_in_adc " << signal_in_adc;
  }
  return signal_in_adc;
}
