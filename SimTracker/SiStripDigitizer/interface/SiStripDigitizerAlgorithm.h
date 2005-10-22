#ifndef SiStripDigitizerAlgorithm_h
#define SiStripDigitizerAlgorithm_h


#include <string>
#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/TrackerGeomFromDetUnits.h"
#include "SimGeneral/Generators/interface/GaussianTailNoiseGenerator.h"
class SiStripDigitizerAlgorithm 
{
 public:
  
  SiStripDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~SiStripDigitizerAlgorithm();
  
  //run the algorithm to digitize a single det
  void run(const std::vector<PSimHit*> &input,StripDigiCollection &output,StripGeomDetUnit *stripdet);


 private:
  

  // Define the internal classes
  //
  // Define a class for 3D ionization points and energy
  //
  /**
   * Internal use only.
   */
  class EnergyDepositUnit{
  public:
    EnergyDepositUnit(): _energy(0),_position(0,0,0){}
    EnergyDepositUnit(float energy,float x, float y, float z):
      _energy(energy),_position(x,y,z){}
    EnergyDepositUnit(float energy, Local3DPoint position):
      _energy(energy),_position(position){}
    float x() const{return _position.x();}
    float y() const{return _position.y();}
    float z() const{return _position.z();}
    float energy() const { return _energy;}
  private:
    float _energy;
    Local3DPoint _position;
  };

  //
// define class to store signals on the collection surface
//
  /**
   * Internal use only.
   */


 class SignalPoint {
  public:
    SignalPoint() : _pos(0,0), _time(0), _amplitude(0), 
     _sigma_x(1.), _sigma_y(1.), _hitp(0) {}
   
   SignalPoint( float x, float y, float sigma_x, float sigma_y,
		float t, float a=1.0) :
   _pos(x,y), _time(t), _amplitude(a), _sigma_x(sigma_x), 
     _sigma_y(sigma_y), _hitp(0) {}

   SignalPoint( float x, float y, float sigma_x, float sigma_y,
		float t, const PSimHit& hit, float a=1.0) :
   _pos(x,y), _time(t), _amplitude(a), _sigma_x(sigma_x), 
     _sigma_y(sigma_y),_hitp(&hit) {}

   const LocalPoint& position() const { return _pos;}
   float x()         const { return _pos.x();}
   float y()         const { return _pos.y();}
   float sigma_x()   const { return _sigma_x;}
   float sigma_y()   const { return _sigma_y;}
   float time()      const { return _time;}
   float amplitude() const { return _amplitude;}
   const PSimHit& hit()           { return *_hitp;}
   SignalPoint& set_amplitude( float amp) { _amplitude = amp; return *this;}

  private:
   LocalPoint         _pos;
   float              _time;
   float              _amplitude;
   float              _sigma_x;   // gaussian sigma in the x direction (cm)
   float              _sigma_y;   //    "       "          y direction (cm) */
   const PSimHit*   _hitp;
 };


//
// definition class
//
  /**
   * Internal use only.
   */

  class Amplitude {
  public:
    Amplitude() : _amp(0.0) { _hits.reserve(1);}
    Amplitude( float amp, const PSimHit* hitp) :
      _amp(amp), _hits(1, hitp) {}

    // can be used as a float by convers.
    operator float() const { return _amp;}

    const std::vector<const PSimHit*>& hits() { return _hits;}

    void operator+=( const Amplitude& other) {
      _amp += other._amp;
      _hits.insert( _hits.end(), other._hits.begin(), other._hits.end());
    }

    void operator+=( const float& amp) {
      _amp += amp;
    }
   
    void set (const float amplitude) {  // Used to reset the amplitude
      _amp = amplitude;
    }

  private:
    float _amp;
    std::vector<const PSimHit*> _hits;

  };  // end class Amplitude




 private:

  edm::ParameterSet conf_;
  //external parameters 
  //-- primary ionization
  int    NumberOfSegments; // =20 does not work ;
  // go from Geant energy GeV to number of electrons
  float GeVperElectron; // 3.7E-09 

  //-- drift
  float Sigma0; //=0.0007  // Charge diffusion in microns for 300 micron Si
  float Dist300;  //=0.0300  // Define 300microns for normalization 

  //-- induce_signal
  float ClusterWidth;       // Gaussian charge cutoff width in sigma units
  //-- make_digis 
  float theElectronPerADC;     // Gain, number of electrons per adc count.
  int theAdcFullScale;         // Saturation count, 255=8bit.
  float theNoiseInElectrons;   // Noise (RMS) in units of electrons.
  float theStripThreshold;     // Strip threshold in units of noise.
  float theStripThresholdInE;  // Strip noise in electorns.
  float theTofCut;             // Cut on the particle TOF
  //-- add_noise
  bool addNoise;
  bool addNoisyStrips;
  bool fluctuateCharge;
  bool addStripInefficiency;
  //-- strip efficiency
  bool stripInefficiency;      // Switch on strip ineffciency
  int  theStripLuminosity;        // luminosity for inefficiency, 0,1,10


  int theStripsInChip;           // num of strips per APV (for strip ineff.)

  int numStrips; // number of strips in a module (detUnit)
  float moduleThickness; // sensor thickness 
  int digis; 
  GeomDetType::SubDetector stripPart;            // is it barrel on forward
  const StripGeomDetUnit* _detp;
   StripTopology* topol;

  std::vector<StripDigi> internal_coll; //empty vector of StripDigi used in digitize


  float StripEff;
  float StripChipEff;
  float StripEfficiency;
  float StripChipEfficiency;
  float theStripEfficiency[3];     // Single strip effciency
  float theStripChipEfficiency[3]; // APV efficiency
 
  //-- calibration smearing
  bool doMissCalibrate;         // Switch on the calibration smearing
  float theGainSmearing;        // The sigma of the gain fluctuation (around 1)
  float theOffsetSmearing;      // The sigma of the offset fluct. (around 0)

  // The PDTable
  //HepPDTable *particleTable;

 //-- charge fluctuation
  double tMax;  // The delta production cut, should be as in OSCAR = 30keV
                //                                           cmsim = 100keV
  // The eloss fluctuation class from G4. Is the right place? 
  SiG4UniversalFluctuation fluctuate; //
  GaussianTailNoiseGenerator* theNoiser; //
  std::vector<const PSimHit*> ss;
 
 std::vector<EnergyDepositUnit> _ionization_points;
  std::vector<SignalPoint> _collection_points;

  typedef std::map< int, Amplitude, std::less<int> >   signal_map_type;  // from
  typedef signal_map_type::iterator          signal_map_iterator; //Digi.Skel.
  signal_map_type     _signal;       // from Digi.Skel.
  
 //-- additional member functions


  void primary_ionization( const PSimHit& hit);
  std::vector<StripDigi> digitize(StripGeomDetUnit *det);
  void drift(const PSimHit& hit);
  void induce_signal( const PSimHit& hit);
  void fluctuateEloss(int particleId, float momentum, float eloss, 
		      float length, int NumberOfSegments,
		      float elossVector[]);
  void add_noise();
  void make_digis();
  void strip_inefficiency();
  float missCalibrate(float amp) const;  


};

#endif
