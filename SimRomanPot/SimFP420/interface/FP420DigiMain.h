#ifndef FP420DigiMain_h
#define FP420DigiMain_h

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/CDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/ChargeDividerFP420.h"

#include "SimRomanPot/SimFP420/interface/GaussNoiseProducerFP420.h"
#include "SimRomanPot/SimFP420/interface/GaussNoiseFP420.h"

#include "SimRomanPot/SimFP420/interface/ZeroSuppressFP420.h"
#include "SimRomanPot/SimFP420/interface/DigiConverterFP420.h"
//#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include <iostream>
#include <vector>



  ////////////////////////////////////////////////////////////////////
class FP420DigiMain { 
  //       interface   interface    interface:
public:

  typedef std::map<int, float, std::less<int> > hit_map_type;
  typedef float Amplitude;

  typedef  DConverterFP420::DigitalMapType DigitalMapType;
  typedef  PileUpFP420::HitToDigisMapType HitToDigisMapType;

  FP420DigiMain(const edm::ParameterSet& conf);
  // FP420DigiMain();
  ~FP420DigiMain();


  // Runs the algorithm
  //  void run(const std::vector<PSimHit*> &input, DigiCollectionFP420 &output,StripGeomDetUnit *det,GlobalVector);
  std::vector <HDigiFP420>  run(const std::vector<PSimHit> &input, G4ThreeVector, unsigned int);
  //vector <HDigiFP420>  run(const std::vector<PSimHit> &input, G4ThreeVector, unsigned int, int);

 private:
  int ndigis; 
  std::vector<short int> adcVec;

  edm::ParameterSet conf_;
  // Const Parameters needed by:
  //-- primary ionization
  int    NumberOfSegments, verbosity, xytype; // 
  // go from Geant energy GeV to number of electrons

  //-- drift
  float Sigma0; //=0.0007  // Charge diffusion in microns for 300 micron Si
  float Thick300;  //=0.0300cm     or = 0.300 mm  - define 300microns for normalization 

  //-- induce_signal
  float ClusterWidth;       // Gaussian charge cutoff width in sigma units
  // Should be rather called CutoffWidth?

  //-- make_digis 
  float theElectronPerADC;     // Gain, number of electrons per adc count.   =  3600 
  double thez420;
  double thezD2;
  double thezD3;
  float ENC;                   // Equivalent noise charge   = 50
  int theAdcFullScale;         // Saturation count, 255=8bit.   = 35000
  float theNoiseInElectrons;   // Noise (RMS) in units of electrons.  = 500
  float theStripThreshold;     // Strip threshold in units of noise.  = 5
  float theStripThresholdInE;  // Strip noise in electorns.  = 2500
  //  bool peakMode; //  = false;
  bool noNoise; //  = false; 
  bool addNoisyPixels;//  = true ;
  bool theApplyTofCut;

  float elossCut;            
  double tofCut;             
  float theThreshold;          // ADC threshold   = 2

  double pitchX;          // pitchX
  double pitchY;          // pitchY
  double pitch;          // pitch automatic
  double pitchXW;          // pitchX
  double pitchYW;          // pitchY
  double pitchW;          // pitch automatic

  double ldriftX;          // ldriftX
  double ldriftY;          // ldriftY
  double ldrift;          // ldrift automatic



  double depletionVoltage; //  =   25.0       !depletion voltage [V]
  double appliedVoltage; //  =   45.0       !bias voltage      [V]
  double chargeMobility; //  = 480.0  !holes    mobility [cm**2/V/sec] p-side;   = 1350.0 !electron mobility - n-side
  double temperature; //  =   
  bool noDiffusion; //  =   true
  double chargeDistributionRMS; //  =  5
  /*
      DH     =   12.3       !diffusion const for holes     [cm**2/sec]
      DE     =   34.6       !diffusion const for electrons [cm**2/sec]

      TGAP   = MUH*BFIELD   !tangent of Lorentz angle for holes
      TGAN   = MUE*BFIELD   !tangent of Lorentz angle for electrons
      W      =    0.0036    !average deposited energy per e-h pair [keV]
      CMB    =    1.6E-19   !electron charge [Coulombs]
      CAP    =   10.0E-15
      GAINP  = 3588.7
      GAINN  = 3594.8
*/
              ChargeDrifterFP420* theCDrifterFP420;
	      /*
		        interface/
		ChargeDrifterFP420.h
		CDrifterFP420.h
		EnergySegmentFP420.h
		AmplitudeSegmentFP420.h
	   	        src/
		ChargeDrifterFP420.cc
	      */
              CDividerFP420* theCDividerFP420;
	      /*
		        interface/
			CDividerFP420.h
			EnergySegmentFP420.h
	      */
///////////	      ChargeDividerFP420* theChargeDividerFP420;
	      /*
		        interface/
			ChargeDividerFP420.h
			LandauFP420.h
	   	        src/
			ChargeDividerFP420.cc
			LandauFP420.cc
	      */
///////////	        IChargeFP420* theIChargeFP420;
	      /*
		          interface/
			  IChargeFP420.h            
			  InduceChargeFP420.h
			  src/
			  InduceChargeFP420.cc
	      */

  GaussNoiseFP420* theGNoiseFP420;
  PileUpFP420* thePileUpFP420;
  HitDigitizerFP420* theHitDigitizerFP420;
  ZeroSuppressFP420* theZSuppressFP420;
  DigiConverterFP420* theDConverterFP420;

  int theStripsInChip;           // num of columns per APV (for strip ineff.)

  int numStripsX;    // number of Xstrips in the module
  int numStripsY;    // number of Ystrips in the module
  int numStrips;    // number of strips in the module
  int numStripsXW;    // number of Xstrips in the module
  int numStripsYW;    // number of Ystrips in the module
  int numStripsW;    // number of strips in the module

  //  int numStripsMax;    // max number of strips in the module
  float moduleThickness; // plate thickness 

  FP420NumberingScheme * theFP420NumberingScheme;

  void push_digis(const DigitalMapType&,
                  const HitToDigisMapType&,
                  const PileUpFP420::signal_map_type&
                  );
                                                                                                  
  //-- calibration smearing
  bool doMissCalibrate;         // Switch on the calibration smearing
  float theGainSmearing;        // The sigma of the gain fluctuation (around 1)
  float theOffsetSmearing;      // The sigma of the offset fluct. (around 0)
                                                                                                  
  // The PDTable
//  HepPDTable *particleTable;
                                                                                                  
  //-- charge fluctuation
  double tMax;  // The delta production cut, should be as in OSCAR = 30keV
                //                                           cmsim = 100keV
  // The eloss fluctuation class from G4. Is the right place?
  LandauFP420 fluctuate; //
  GaussNoiseProducerFP420* theNoiser; //
  std::vector<const PSimHit*> ss;  // ss - pointers to hit info of PSimHit
                                                                                                  
  void fluctuateEloss(int particleId, float momentum, float eloss,
                      float length, int NumberOfSegments,
                      float elossVector[]);
                                                                                                  
//  std::vector<HDigiFP420> internal_coll; //empty vector of HDigiFP420 used in digitize // AZ


  std::vector<HDigiFP420> digis;


//  std::vector<HDigiFP420> digitize(StripGeomDetUnit *det);    // AZ
//  int rn0,pn0,sn0;                                                                                          
                                                                                                  
};
                                                                                                  


// end of interface
  ////////////////////////////////////////////////////////////////////

#endif

