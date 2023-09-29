#ifndef SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizerAlgorithm_h
#define SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizerAlgorithm_h

#include <map>
#include <memory>
#include <vector>

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include "SimTracker/Common/interface/DigitizerUtility.h"
#include "SimTracker/Common/interface/SiPixelChargeReweightingAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"

// Units and Constants
#include "DataFormats/Math/interface/CMSUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

// forward declarations
// For the random numbers
namespace CLHEP {
  class HepRandomEngine;
  class RandGaussQ;
  class RandFlat;
}  // namespace CLHEP

class DetId;
class GaussianTailNoiseGenerator;
class SiG4UniversalFluctuation;
class SiPixelGainCalibrationOfflineSimService;
class SiPixelLorentzAngle;
class SiPixelQuality;
class SiPhase2OuterTrackerLorentzAngle;
class TrackerGeometry;
class TrackerTopology;
class SiPixelChargeReweightingAlgorithm;

// REMEMBER CMS conventions:
// -- Energy: GeV
// -- momentum: GeV/c
// -- mass: GeV/c^2
// -- Distance, position: cm
// -- Time: ns
// -- Angles: radian
// Some constants in convenient units
constexpr double c_cm_ns = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
constexpr double c_inv = 1.0 / c_cm_ns;

class Phase2TrackerDigitizerAlgorithm {
public:
  Phase2TrackerDigitizerAlgorithm(const edm::ParameterSet& conf_common,
                                  const edm::ParameterSet& conf_specific,
                                  edm::ConsumesCollector iC);
  virtual ~Phase2TrackerDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  virtual void init(const edm::EventSetup& es) = 0;
  virtual void initializeEvent(CLHEP::HepRandomEngine& eng);

  // run the algorithm to digitize a single det
  virtual void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                                 const std::vector<PSimHit>::const_iterator inputEnd,
                                 const size_t inputBeginGlobalIndex,
                                 const uint32_t tofBin,
                                 const Phase2TrackerGeomDetUnit* pixdet,
                                 const GlobalVector& bfield);
  virtual void digitize(const Phase2TrackerGeomDetUnit* pixdet,
                        std::map<int, digitizerUtility::DigiSimInfo>& digi_map,
                        const TrackerTopology* tTopo);
  virtual bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) const { return true; }
  virtual bool isAboveThreshold(const digitizerUtility::SimHitInfo* hitInfo, float charge, float thr) const {
    return true;
  }

  // For premixing
  void loadAccumulator(uint32_t detId, const std::map<int, float>& accumulator);

protected:
  // Accessing Inner Tracker Lorentz angle from DB:
  const SiPixelLorentzAngle* siPixelLorentzAngle_;

  // Accessing Outer Tracker Lorentz angle from DB:
  const SiPhase2OuterTrackerLorentzAngle* siPhase2OTLorentzAngle_;

  // Accessing Dead pixel modules from DB:
  const SiPixelQuality* siPixelBadModule_;

  // Accessing Map and Geom:
  const TrackerGeometry* geom_;
  struct SubdetEfficiencies {
    SubdetEfficiencies(const edm::ParameterSet& conf);
    std::vector<double> barrel_efficiencies;
    std::vector<double> endcap_efficiencies;
  };

  // Internal type aliases
  using signal_map_type = std::map<int, digitizerUtility::Ph2Amplitude, std::less<int> >;
  using signalMaps = std::map<uint32_t, signal_map_type>;
  using Frame = GloballyPositioned<double>;
  using Parameters = std::vector<edm::ParameterSet>;

  // Contains the accumulated hit info.
  signalMaps _signal;

  const bool makeDigiSimLinks_;

  const bool use_ineff_from_db_;
  const bool use_module_killing_;   // remove or not the dead pixel modules
  const bool use_deadmodule_DB_;    // if we want to get dead pixel modules from the DataBase.
  const bool use_LorentzAngle_DB_;  // if we want to get Lorentz angle from the DataBase.

  const Parameters deadModules_;

  // Variables
  // external parameters
  // go from Geant energy GeV to number of electrons
  const float GeVperElectron_;  // 3.7E-09

  //-- drift
  const bool alpha2Order_;  // Switch on/off of E.B effect
  const bool addXtalk_;
  const float interstripCoupling_;
  const float Sigma0_;      //=0.0007  // Charge diffusion in microns for 300 micron Si
  const float SigmaCoeff_;  // delta in the diffusion across the strip pitch

  //-- induce_signal
  const float clusterWidth_;  // Gaussian charge cutoff width in sigma units

  //-- make_digis
  const int thePhase2ReadoutMode_;   //  Flag to decide readout mode (digital/Analog dual slope etc.)
  const float theElectronPerADC_;    // Gain, number of electrons per adc count.
  const int theAdcFullScale_;        // Saturation count, 255=8bit.
  const float theNoiseInElectrons_;  // Noise (RMS) in units of electrons.
  const float theReadoutNoise_;      // Noise of the readount chain in elec,

  // inludes DCOL-Amp,TBM-Amp, Alt, AOH,OptRec.
  const float theThresholdInE_Endcap_;  // threshold in electrons Endcap.
  const float theThresholdInE_Barrel_;  // threshold in electrons Barrel.

  const double theThresholdSmearing_Endcap_;
  const double theThresholdSmearing_Barrel_;

  const double theHIPThresholdInE_Endcap_;
  const double theHIPThresholdInE_Barrel_;

  const float theTofLowerCut_;                  // Cut on the particle TOF
  const float theTofUpperCut_;                  // Cut on the particle TOF
  const float tanLorentzAnglePerTesla_Endcap_;  //FPix Lorentz angle tangent per Tesla
  const float tanLorentzAnglePerTesla_Barrel_;  //BPix Lorentz angle tangent per Tesla

  // -- add_noise
  const bool addNoise_;
  const bool addNoisyPixels_;
  const bool fluctuateCharge_;

  //-- pixel efficiency
  const bool addPixelInefficiency_;  // bool to read in inefficiencies

  const bool addThresholdSmearing_;

  // pseudoRadDamage
  const double pseudoRadDamage_;        // Decrease the amount off freed charge that reaches the collector
  const double pseudoRadDamageRadius_;  // Only apply pseudoRadDamage to pixels with radius<=pseudoRadDamageRadius

  // charge reweighting
  const bool useChargeReweighting_;
  // access 2D templates from DB. Only gets initialized if useChargeReweighting_ is set to true
  const std::unique_ptr<SiPixelChargeReweightingAlgorithm> theSiPixelChargeReweightingAlgorithm_;

  // The PDTable
  // HepPDTable *particleTable;
  // ParticleDataTable *particleTable;

  //-- charge fluctuation
  const double tMax_;  // The delta production cut, should be as in OSCAR = 30keV

  // Bad Pixels to be killed
  Parameters badPixels_;

  // The eloss fluctuation class from G4. Is the right place?
  const std::unique_ptr<SiG4UniversalFluctuation> fluctuate_;  // make a pointer
  const std::unique_ptr<GaussianTailNoiseGenerator> theNoiser_;

  //-- additional member functions
  // Private methods
  virtual std::vector<digitizerUtility::EnergyDepositUnit> primary_ionization(const PSimHit& hit) const;
  virtual std::vector<digitizerUtility::SignalPoint> drift(
      const PSimHit& hit,
      const Phase2TrackerGeomDetUnit* pixdet,
      const GlobalVector& bfield,
      const std::vector<digitizerUtility::EnergyDepositUnit>& ionization_points) const;
  virtual void induce_signal(std::vector<PSimHit>::const_iterator inputBegin,
                             const PSimHit& hit,
                             const size_t hitIndex,
                             const size_t firstHitIndex,
                             const uint32_t tofBin,
                             const Phase2TrackerGeomDetUnit* pixdet,
                             const std::vector<digitizerUtility::SignalPoint>& collection_points);
  virtual std::vector<float> fluctuateEloss(
      int particleId, float momentum, float eloss, float length, int NumberOfSegments) const;
  virtual void add_noise(const Phase2TrackerGeomDetUnit* pixdet);
  virtual void add_cross_talk(const Phase2TrackerGeomDetUnit* pixdet);
  virtual void add_noisy_cells(const Phase2TrackerGeomDetUnit* pixdet, float thePixelThreshold);
  virtual void pixel_inefficiency(const SubdetEfficiencies& eff,
                                  const Phase2TrackerGeomDetUnit* pixdet,
                                  const TrackerTopology* tTopo);

  virtual void pixel_inefficiency_db(uint32_t detID);

  // access to the gain calibration payloads in the db. Only gets initialized if check_dead_pixels_ is set to true.
  const std::unique_ptr<SiPixelGainCalibrationOfflineSimService> theSiPixelGainCalibrationService_;

  LocalVector driftDirection(const Phase2TrackerGeomDetUnit* pixdet,
                             const GlobalVector& bfield,
                             const DetId& detId) const;

  // remove dead modules using the list in the configuration file PixelDigi_cfi.py
  virtual void module_killing_conf(uint32_t detID);
  // remove dead modules uisng the list in the DB
  virtual void module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) = 0;

  const SubdetEfficiencies subdetEfficiencies_;
  float calcQ(float x);

  // For random numbers
  std::unique_ptr<CLHEP::RandGaussQ> gaussDistribution_;

  // Threshold gaussian smearing:
  std::unique_ptr<CLHEP::RandGaussQ> smearedThreshold_Endcap_;
  std::unique_ptr<CLHEP::RandGaussQ> smearedThreshold_Barrel_;

  //for engine passed into the constructor from Digitizer
  CLHEP::HepRandomEngine* rengine_;

  // convert signal in electrons to ADC counts
  int convertSignalToAdc(uint32_t detID, float signal_in_elec, float threshold);

  bool pixelFlag_;
};
#endif
