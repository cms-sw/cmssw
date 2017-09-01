#ifndef SimG4CMS_DreamSD_h
#define SimG4CMS_DreamSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#include "G4PhysicsOrderedFreeVector.hh"

#include <map>

const int MAXPHOTONS = 500; // Maximum number of photons we can store

class G4LogicalVolume;

class TTree;

class DreamSD : public CaloSD {

public:    

  DreamSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	  edm::ParameterSet const &, const SimTrackManager*);
  ~DreamSD() override {}
  bool   ProcessHits(G4Step * step,G4TouchableHistory * tHistory) override;
  uint32_t setDetUnitId(const G4Step*) override;

protected:

  G4bool getStepInfo(const G4Step* aStep) override;
  void   initRun() override;

private:    

  typedef std::pair<double,double> Doubles;
  typedef std::map<G4LogicalVolume*,Doubles> DimensionMap;

  void     initMap(const G4String&, const DDCompactView &);
  double   curve_LY(); 
  double   crystalLength() const;
  double   crystalWidth() const;

  /// Returns the total energy due to Cherenkov radiation
  double         cherenkovDeposit_(const G4Step* aStep );
  /// Returns average number of photons created by track
  double getAverageNumberOfPhotons_(double charge,
				    double beta,
				    const G4Material* aMaterial,
				    const G4MaterialPropertyVector* rIndex );
  /// Returns energy deposit for a given photon
  double getPhotonEnergyDeposit_( const G4ParticleMomentum& p, 
				  const G4ThreeVector& x);
  /// Sets material properties at run-time...
  bool setPbWO2MaterialProperties_(G4Material* aMaterial );

  bool         useBirk, doCherenkov_, readBothSide_;
  double       birk1, birk2, birk3;
  double       slopeLY;
  DimensionMap xtalLMap; // Store length and width

  int          side;

  /// Table of Cherenkov angle integrals vs photon momentum
  std::unique_ptr<G4PhysicsOrderedFreeVector> chAngleIntegrals_;
  G4MaterialPropertiesTable*                materialPropertiesTable;
  // Histogramming
  TTree* ntuple_;
  int nphotons_;
  float px_[MAXPHOTONS],py_[MAXPHOTONS],pz_[MAXPHOTONS];
  float x_[MAXPHOTONS],y_[MAXPHOTONS],z_[MAXPHOTONS];

};

#endif // DreamSD_h
