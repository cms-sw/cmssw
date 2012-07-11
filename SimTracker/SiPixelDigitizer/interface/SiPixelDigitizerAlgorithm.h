#ifndef SiPixelDigitizerAlgorithm_h
#define SiPixelDigitizerAlgorithm_h


#include <string>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

//#include "SimGeneral/HepPDT/interface/HepPDTable.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomFromDetUnits.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
//#include "DataFormats/GeometrySurface/interface/TkRotation.h"
//#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
//#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"

// pixel gain payload access (offline version for Simulation)
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

// Accessing Pixel Lorentz Angle from the DB:
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"

// Accessing Pixel dead modules from the DB:
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


// For the random numbers
namespace CLHEP {
  class HepRandomEngine;
  class RandGaussQ;
  class RandFlat;
}

class SiPixelDigitizerAlgorithm  {
 public:
  
  SiPixelDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  ~SiPixelDigitizerAlgorithm();
  
  //run the algorithm to digitize a single det
  edm::DetSet<PixelDigi>::collection_type  
    run(const std::vector<PSimHit> &input,PixelGeomDetUnit *pixdet,GlobalVector);

   //
  std::vector<PixelDigiSimLink> make_link() {
    return link_coll; }
  void init(const edm::EventSetup& es);
  void fillDeadModules(const edm::EventSetup& es);
  void fillLorentzAngle(const edm::EventSetup& es);
  void fillMapandGeom(const edm::EventSetup& es);


 private:
  
  //Accessing Lorentz angle from DB:
  edm::ESHandle<SiPixelLorentzAngle> SiPixelLorentzAngle_;

  //Accessing Dead pixel modules from DB:
  edm::ESHandle<SiPixelQuality> SiPixelBadModule_;

  //Accessing Map and Geom:
  edm::ESHandle<SiPixelFedCablingMap> map_;
  edm::ESHandle<TrackerGeometry> geom_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters DeadModules;

  // Define internal classes
  
  // Define a class to hold the calibration parameters per pixel
  // Internal
  class CalParameters {
  public:
    float p0;
    float p1;
    float p2;
    float p3;
  };
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
    Amplitude( float amp, const PSimHit* hitp, float frac) :
      _amp(amp), _hits(1, hitp), _frac(1,frac) {

    //in case of digi from noisypixels
      //the MC information are removed 
      if (_frac[0]<-0.5) {
	_frac.pop_back();
	_hits.pop_back();
     }

    }

    // can be used as a float by convers.
    operator float() const { return _amp;}
    float ampl() const {return _amp;}
    std::vector<float> individualampl() const {return _frac;}
    const std::vector<const PSimHit*>& hits() { return _hits;}

    void operator+=( const Amplitude& other) {
      _amp += other._amp;
      //in case of contribution of noise to the digi
      //the MC information are removed 
      if (other._frac[0]>-0.5){
	_hits.insert( _hits.end(), other._hits.begin(), other._hits.end());
	_frac.insert(_frac.end(), other._frac.begin(), other._frac.end());
      }
   }

    void operator+=( const float& amp) {
      _amp += amp;
    }
   
    void set (const float amplitude) {  // Used to reset the amplitude
      _amp = amplitude;
    }
/*     void setind (const float indamplitude) {  // Used to reset the amplitude */
/*       _frac = idamplitude; */
/*     } */
  private:
    float _amp;
    std::vector<const PSimHit*> _hits;
    std::vector<float> _frac;
  };  // end class Amplitude


 private:

    // Internal typedefs
    typedef std::map< int, Amplitude, std::less<int> >   signal_map_type;  // from
    typedef signal_map_type::iterator          signal_map_iterator; //Digi.Skel.  
    typedef std::map<unsigned int, std::vector<float>,std::less<unsigned int> > 
      simlink_map;
    typedef GloballyPositioned<double>      Frame;

    // Variables 
    edm::ParameterSet conf_;
    //external parameters 
    //-- primary ionization
    int    NumberOfSegments; // =20 does not work ;
    // go from Geant energy GeV to number of electrons
    float GeVperElectron; // 3.7E-09 
    
    //-- drift
    float Sigma0; //=0.0007  // Charge diffusion in microns for 300 micron Si
    float Dist300;  //=0.0300  // Define 300microns for normalization 
    bool alpha2Order;          // Switch on/off of E.B effect 


 
    //-- induce_signal
    float ClusterWidth;       // Gaussian charge cutoff width in sigma units
    //-- make_digis 
    float theElectronPerADC;     // Gain, number of electrons per adc count.
    int theAdcFullScale;         // Saturation count, 255=8bit.
    float theNoiseInElectrons;   // Noise (RMS) in units of electrons.
    float theReadoutNoise;       // Noise of the readount chain in elec,
                                 //inludes DCOL-Amp,TBM-Amp, Alt, AOH,OptRec.

    float theSmearedChargeRMS;

    float thePixelThreshold;     // Pixel threshold in units of noise.

    float thePixelThresholdInE;  // Pixel noise in electrons.

    float theThresholdInE_FPix;  // Pixel threshold in electrons FPix.
    float theThresholdInE_BPix;  // Pixel threshold in electrons BPix.
    //Carlotta  //--CPC: someone change
    float theThresholdInE_BPix_L1;

    double theThresholdSmearing_FPix;
    double theThresholdSmearing_BPix;
    //Carlotta: inly if you also want different smearing il L1
    //double theThresholdSmearing_BPix_L1;

    double electronsPerVCAL;          // for electrons - VCAL conversion
    double electronsPerVCAL_Offset;   // in misscalibrate()

    float theTofLowerCut;             // Cut on the particle TOF
    float theTofUpperCut;             // Cut on the particle TOF
    float tanLorentzAnglePerTesla_FPix;   //FPix Lorentz angle tangent per Tesla
    float tanLorentzAnglePerTesla_BPix;   //BPix Lorentz angle tangent per Tesla

    float FPix_p0;
    float FPix_p1;
    float FPix_p2;
    float FPix_p3;
    float BPix_p0;
    float BPix_p1;
    float BPix_p2;
    float BPix_p3;


    //-- add_noise
    bool addNoise;
    bool addChargeVCALSmearing;
    bool addNoisyPixels;
    bool fluctuateCharge;
    bool addPixelInefficiency;
    //-- pixel efficiency
    bool pixelInefficiency;      // Switch on pixel ineffciency
    int  thePixelLuminosity;        // luminosity for inefficiency, 0,1,10

    bool addThresholdSmearing;
        
    int theColsInChip;           // num of columns per ROC (for pix ineff.)
    int theRowsInChip;           // num of rows per ROC
    
    int numColumns; // number of pixel columns in a module (detUnit)
    int numRows;    // number          rows
    float moduleThickness; // sensor thickness 
    //  int digis; 
    const PixelGeomDetUnit* _detp;
    uint32_t detID;     // Det id
    

    std::vector<PSimHit> _PixelHits; //cache
    const PixelTopology* topol;
    
    std::vector<PixelDigi> internal_coll; //empty vector of PixelDigi used in digitize

    std::vector<PixelDigiSimLink> link_coll;
    GlobalVector _bfield;
    
    float PixelEff;
    float PixelColEff;
    float PixelChipEff;
    float PixelEfficiency;
    float PixelColEfficiency;
    float PixelChipEfficiency;
    float thePixelEfficiency[6];     // Single pixel effciency
    float thePixelColEfficiency[6];  // Column effciency
    float thePixelChipEfficiency[6]; // ROC efficiency
    
    //-- calibration smearing
    bool doMissCalibrate;         // Switch on the calibration smearing
    float theGainSmearing;        // The sigma of the gain fluctuation (around 1)
    float theOffsetSmearing;      // The sigma of the offset fluct. (around 0)
    

    // The PDTable
    //HepPDTable *particleTable;
    //ParticleDataTable *particleTable;

    //-- charge fluctuation
    double tMax;  // The delta production cut, should be as in OSCAR = 30keV
    //                                           cmsim = 100keV
    // The eloss fluctuation class from G4. Is the right place? 
    SiG4UniversalFluctuation * fluctuate;   // make a pointer 
    GaussianTailNoiseGenerator * theNoiser; //




    PixelIndices * pIndexConverter;         // Pointer to the index converter 
   
    std::vector<EnergyDepositUnit> _ionization_points;
    std::vector<SignalPoint> _collection_points;
    
    simlink_map simi;
    signal_map_type     _signal;       // from Digi.Skel.

    // To store calibration constants
    std::map<int,CalParameters,std::less<int> > calmap;


    //-- additional member functions    
    // Private methods
    void primary_ionization( const PSimHit& hit);
    std::vector<PixelDigi> digitize(PixelGeomDetUnit *det);
    void drift(const PSimHit& hit);
    void induce_signal( const PSimHit& hit);
    void fluctuateEloss(int particleId, float momentum, float eloss, 
			float length, int NumberOfSegments,
			float elossVector[]);
    void add_noise();
    void make_digis();
    void pixel_inefficiency();
    bool use_ineff_from_db_;

    bool use_module_killing_; // remove or not the dead pixel modules
    bool use_deadmodule_DB_; // if we want to get dead pixel modules from the DataBase.
    bool use_LorentzAngle_DB_; // if we want to get Lorentz angle from the DataBase.

    void pixel_inefficiency_db(); 
       // access to the gain calibration payloads in the db. Only gets initialized if check_dead_pixels_ is set to true.
    SiPixelGainCalibrationOfflineSimService * theSiPixelGainCalibrationService_;    
    float missCalibrate(int col, int row, float amp) const;  
    LocalVector DriftDirection();

    void module_killing_conf(); // remove dead modules using the list in the configuration file PixelDigi_cfi.py
    void module_killing_DB();  // remove dead modules uisng the list in the DB

   // For random numbers
    CLHEP::HepRandomEngine& rndEngine;

    CLHEP::RandFlat *flatDistribution_;
    CLHEP::RandGaussQ *gaussDistribution_;
    CLHEP::RandGaussQ *gaussDistributionVCALNoise_;

    
    // Threshold gaussian smearing:
    CLHEP::RandGaussQ *smearedThreshold_FPix_;
    CLHEP::RandGaussQ *smearedThreshold_BPix_;
    //Carlotta
    CLHEP::RandGaussQ *smearedThreshold_BPix_L1_;
    
    CLHEP::RandGaussQ *smearedChargeDistribution_ ;
    
    // the random generator
    CLHEP::RandGaussQ* theGaussianDistribution;


};

#endif
