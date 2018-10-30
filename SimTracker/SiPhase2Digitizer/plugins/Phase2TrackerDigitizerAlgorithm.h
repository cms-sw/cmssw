#ifndef __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizerAlgorithm_h
#define __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizerAlgorithm_h

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/Common/interface/SimHitInfoForLinks.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "SimTracker/SiPhase2Digitizer/plugins/DigitizerUtility.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"

// forward declarations
// For the random numbers
namespace CLHEP {
  class HepRandomEngine;
  class RandGaussQ;
  class RandFlat;
}

namespace edm {
  class EventSetup;
  class ParameterSet;
}

class DetId;
class GaussianTailNoiseGenerator;
class SiG4UniversalFluctuation;
class SiPixelFedCablingMap;
class SiPixelGainCalibrationOfflineSimService;
class SiPixelLorentzAngle;
class SiPixelQuality;
class TrackerGeometry;
class TrackerTopology;

class Phase2TrackerDigitizerAlgorithm  {
 public:
  Phase2TrackerDigitizerAlgorithm(const edm::ParameterSet& conf_common, const edm::ParameterSet& conf_specific);
  virtual ~Phase2TrackerDigitizerAlgorithm(); 

  // initialization that cannot be done in the constructor
  virtual void init(const edm::EventSetup& es) = 0;
  virtual void initializeEvent(CLHEP::HepRandomEngine& eng);
  // run the algorithm to digitize a single det
  virtual void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
				 const std::vector<PSimHit>::const_iterator inputEnd,
				 const size_t inputBeginGlobalIndex,
				 const unsigned int tofBin,
				 const Phase2TrackerGeomDetUnit* pixdet,
				 const GlobalVector& bfield) = 0;
 virtual void digitize(const Phase2TrackerGeomDetUnit* pixdet,
		       std::map<int, DigitizerUtility::DigiSimInfo>& digi_map,
		       const TrackerTopology* tTopo);

  // For premixing
  void loadAccumulator(unsigned int detId, const std::map<int, float>& accumulator);
 protected:
  // Accessing Lorentz angle from DB:
  edm::ESHandle<SiPixelLorentzAngle> SiPixelLorentzAngle_;

  // Accessing Dead pixel modules from DB:
  edm::ESHandle<SiPixelQuality> SiPixelBadModule_;

  // Accessing Map and Geom:
  edm::ESHandle<SiPixelFedCablingMap> map_;
  edm::ESHandle<TrackerGeometry> geom_;
  struct SubdetEfficiencies {
    SubdetEfficiencies(const edm::ParameterSet& conf);
    std::vector<double> barrel_efficiencies;
    std::vector<double> endcap_efficiencies;
  };
  
  // Internal type aliases 
  using signal_map_type           = std::map<int, DigitizerUtility::Amplitude, std::less<int> >;  // from Digi.Skel.
  using signal_map_iterator       = signal_map_type::iterator; // from Digi.Skel.  
  using signal_map_const_iterator = signal_map_type::const_iterator; // from Digi.Skel.  
  using simlink_map               = std::map<unsigned int, std::vector<float>,std::less<unsigned int> > ;
  using signalMaps                = std::map<uint32_t, signal_map_type> ;
  using Frame                     = GloballyPositioned<double> ;
  using Parameters                = std::vector<edm::ParameterSet> ;
  
  // Contains the accumulated hit info.
  signalMaps _signal;

  const bool makeDigiSimLinks_;

  const bool use_ineff_from_db_;
  const bool use_module_killing_; // remove or not the dead pixel modules
  const bool use_deadmodule_DB_; // if we want to get dead pixel modules from the DataBase.
  const bool use_LorentzAngle_DB_; // if we want to get Lorentz angle from the DataBase.
  
  const Parameters DeadModules;

  // Variables 
  // external parameters 
  // go from Geant energy GeV to number of electrons
  const float GeVperElectron; // 3.7E-09 
    
  //-- drift
  const bool alpha2Order;          // Switch on/off of E.B effect 
  const bool addXtalk;
  const float interstripCoupling;     
  const float Sigma0; //=0.0007  // Charge diffusion in microns for 300 micron Si
  const float SigmaCoeff; // delta in the diffusion across the strip pitch 
  
  //-- induce_signal
  const float ClusterWidth;       // Gaussian charge cutoff width in sigma units
  
  //-- make_digis 
  const int   thePhase2ReadoutMode;  //  Flag to decide readout mode (digital/Analog dual slope etc.)
  const float theElectronPerADC;     // Gain, number of electrons per adc count.
  const int theAdcFullScale;         // Saturation count, 255=8bit.
  const float theNoiseInElectrons;   // Noise (RMS) in units of electrons.
  const float theReadoutNoise;       // Noise of the readount chain in elec,

  // inludes DCOL-Amp,TBM-Amp, Alt, AOH,OptRec.
  const float theThresholdInE_Endcap;  // threshold in electrons Endcap.
  const float theThresholdInE_Barrel;  // threshold in electrons Barrel.

  const double theThresholdSmearing_Endcap;
  const double theThresholdSmearing_Barrel;

  const double theHIPThresholdInE_Endcap;
  const double theHIPThresholdInE_Barrel;

  const float theTofLowerCut;             // Cut on the particle TOF
  const float theTofUpperCut;             // Cut on the particle TOF
  const float tanLorentzAnglePerTesla_Endcap;   //FPix Lorentz angle tangent per Tesla
  const float tanLorentzAnglePerTesla_Barrel;   //BPix Lorentz angle tangent per Tesla

  // -- add_noise
  const bool addNoise;
  const bool addNoisyPixels;
  const bool fluctuateCharge;

  //-- pixel efficiency
  const bool AddPixelInefficiency;        // bool to read in inefficiencies

  const bool addThresholdSmearing;
    
  // pseudoRadDamage
  const double pseudoRadDamage;       // Decrease the amount off freed charge that reaches the collector
  const double pseudoRadDamageRadius; // Only apply pseudoRadDamage to pixels with radius<=pseudoRadDamageRadius

  // The PDTable
  // HepPDTable *particleTable;
  // ParticleDataTable *particleTable;

  //-- charge fluctuation
  const double tMax;  // The delta production cut, should be as in OSCAR = 30keV

  // Bad Pixels to be killed
  std::vector<edm::ParameterSet> badPixels;

  // The eloss fluctuation class from G4. Is the right place? 
  const std::unique_ptr<SiG4UniversalFluctuation> fluctuate;   // make a pointer
  const std::unique_ptr<GaussianTailNoiseGenerator> theNoiser;

  //-- additional member functions    
  // Private methods
  void primary_ionization( const PSimHit& hit, std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points) const;
  void drift(const PSimHit& hit,
	     const Phase2TrackerGeomDetUnit* pixdet,
	     const GlobalVector& bfield,
	     const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points,
	     std::vector<DigitizerUtility::SignalPoint>& collection_points) const;
  void induce_signal(const PSimHit& hit,
		     const size_t hitIndex,
		     const unsigned int tofBin,
		     const Phase2TrackerGeomDetUnit* pixdet,
		     const std::vector<DigitizerUtility::SignalPoint>& collection_points);
  void fluctuateEloss(int particleId, float momentum, float eloss, 
		      float length, int NumberOfSegments,
		      std::vector<float> & elossVector) const;
  virtual void add_noise(const Phase2TrackerGeomDetUnit* pixdet, float thePixelThreshold);
  virtual void pixel_inefficiency(const SubdetEfficiencies& eff,
				  const Phase2TrackerGeomDetUnit* pixdet,
				  const TrackerTopology* tTopo);
  
  virtual void pixel_inefficiency_db(uint32_t detID);

  // access to the gain calibration payloads in the db. Only gets initialized if check_dead_pixels_ is set to true.
  const std::unique_ptr<SiPixelGainCalibrationOfflineSimService> theSiPixelGainCalibrationService_;    
  LocalVector DriftDirection(const Phase2TrackerGeomDetUnit* pixdet,
			     const GlobalVector& bfield,
			     const DetId& detId) const;
  
  virtual void module_killing_conf(uint32_t detID); // remove dead modules using the list in the configuration file PixelDigi_cfi.py
  virtual void module_killing_DB(uint32_t detID);  // remove dead modules uisng the list in the DB

  const SubdetEfficiencies subdetEfficiencies_;
  
  // For random numbers
  std::unique_ptr<CLHEP::RandGaussQ> gaussDistribution_;
  
  // Threshold gaussian smearing:
  std::unique_ptr<CLHEP::RandGaussQ> smearedThreshold_Endcap_;
  std::unique_ptr<CLHEP::RandGaussQ> smearedThreshold_Barrel_;
  
  //for engine passed into the constructor from Digitizer
  CLHEP::HepRandomEngine* rengine_;   

  // convert signal in electrons to ADC counts
  int convertSignalToAdc(uint32_t detID,float signal_in_elec,float threshold);

  double calcQ(float x) const {
    auto xx = std::min(0.5f * x * x,12.5f);
    return 0.5 * (1.0-std::copysign(std::sqrt(1.f- unsafe_expf<4>(-xx * (1.f + 0.2733f/(1.f + 0.147f * xx)))), x));
  }
  bool pixelFlag;
};
#endif
