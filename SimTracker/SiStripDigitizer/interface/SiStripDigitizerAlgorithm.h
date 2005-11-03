#ifndef SiStripDigitizerAlgorithm_h
#define SiStripDigitizerAlgorithm_h

/** \class SiStripDigitizerAlgorithm
 *
 * SiStripDigitizerAlgorithm converts hits to digis
 *
 * \author Andrea Giammanco
 *
 * \version   1st Version Sep. 29, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialZeroSuppress.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"
#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/TrackerGeomFromDetUnits.h"
#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
using namespace std;

class SiStripDigitizerAlgorithm 
{
 public:

  typedef  SiDigitalConverter::DigitalMapType DigitalMapType;
  typedef  SiPileUpSignals::HitToDigisMapType HitToDigisMapType;
  typedef map< int, float, less<int> > hit_map_type;
  typedef float Amplitude;
  
  SiStripDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~SiStripDigitizerAlgorithm();

  // Runs the algorithm
  //  void run(const PSimHit* input, StripDigiCollection &output);
  void run(const std::vector<PSimHit*> &input, StripDigiCollection &output,StripGeomDetUnit *det);

 private:
  int ndigis; 
  vector<short int> adcVec;

  edm::ParameterSet conf_;
  // Const Parameters needed by:
  //-- primary ionization
  int    NumberOfSegments; // =20 does not work ;
  // go from Geant energy GeV to number of electrons

  //-- drift
  float Sigma0; //=0.0007  // Charge diffusion in microns for 300 micron Si
  float Dist300;  //=0.0300  // Define 300microns for normalization 

  //-- induce_signal
  float ClusterWidth;       // Gaussian charge cutoff width in sigma units
  // Should be rather called CutoffWidth?

  //-- make_digis 
  float theElectronPerADC;     // Gain, number of electrons per adc count.
  float ENC;                   // Equivalent noise charge
  int theAdcFullScale;         // Saturation count, 255=8bit.
  float theNoiseInElectrons;   // Noise (RMS) in units of electrons.
  float theStripThreshold;     // Strip threshold in units of noise.
  float theStripThresholdInE;  // Strip noise in electorns.
  bool peakMode;
  bool noNoise;
  float tofCut;             // Cut on the particle TOF
  float theThreshold;          // ADC threshold

  double depletionVoltage;
  double appliedVoltage;
  double chargeMobility;
  double temperature;
  bool noDiffusion;
  double chargeDistributionRMS;
  SiLinearChargeCollectionDrifter* theSiChargeCollectionDrifter;
  SiChargeDivider* theSiChargeDivider;
  SiGaussianTailNoiseAdder* theSiNoiseAdder;
  SiPileUpSignals* theSiPileUpSignals;
  SiHitDigitizer* theSiHitDigitizer;
  SiTrivialZeroSuppress* theSiZeroSuppress;
  SiTrivialDigitalConverter* theSiDigitalConverter;

  int theStripsInChip;           // num of columns per APV (for strip ineff.)

  int numStrips;    // number of strips in the module
  //  int numStripsMax;    // max number of strips in the module
  float moduleThickness; // sensor thickness 

  //-- add_noise
  /*
  bool addNoise;
  bool addNoisyStrips;
  bool fluctuateCharge;
  */

  void push_digis(StripDigiCollection &,
		  const DigitalMapType&,
		  const HitToDigisMapType&,
		  const SiPileUpSignals::signal_map_type&,
		  unsigned int);
 
  //-- calibration smearing
  bool doMissCalibrate;         // Switch on the calibration smearing
  float theGainSmearing;        // The sigma of the gain fluctuation (around 1)
  float theOffsetSmearing;      // The sigma of the offset fluct. (around 0)

  // The PDTable
  HepPDTable *particleTable;

  //-- charge fluctuation
  double tMax;  // The delta production cut, should be as in OSCAR = 30keV
                //                                           cmsim = 100keV
  // The eloss fluctuation class from G4. Is the right place? 
  SiG4UniversalFluctuation fluctuate; //
  GaussianTailNoiseGenerator* theNoiser; //
  std::vector<const PSimHit*> ss;

  void fluctuateEloss(int particleId, float momentum, float eloss, 
		      float length, int NumberOfSegments,
		      float elossVector[]);
  
  GeomDetType::SubDetector stripPart;            // is it barrel on forward
  const StripGeomDetUnit* _detp;
  StripTopology* topol;

  std::vector<StripDigi> internal_coll; //empty vector of StripDigi used in digitize

  std::vector<StripDigi> digitize(StripGeomDetUnit *det);


};

#endif
