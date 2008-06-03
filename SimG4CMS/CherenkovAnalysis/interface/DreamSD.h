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

  DreamSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~DreamSD() {}
  virtual double                    getEnergyDeposit(G4Step*);
  virtual uint32_t                  setDetUnitId(G4Step*);

private:    

  typedef std::pair<double,double> Doubles;
  typedef std::map<G4LogicalVolume*,Doubles> DimensionMap;

  void   initMap(G4String, const DDCompactView &);
  double curve_LY(G4Step*); 
  const double crystalLength(G4LogicalVolume*) const;
  const double crystalWidth(G4LogicalVolume*) const;

  /// Returns the total energy due to Cherenkov radiation
  double cherenkovDeposit_( G4Step* aStep );
  /// Returns average number of photons created by track
  const double getAverageNumberOfPhotons_( const double charge,
                                           const double beta,
                                           const G4Material* aMaterial,
                                           const G4MaterialPropertyVector* rIndex ) const;
  /// Returns energy deposit for a given photon
  const double getPhotonEnergyDeposit_( const G4ParticleMomentum& p, const G4ThreeVector& x,
                                        const G4Step* aStep );
  /// Sets material properties at run-time...
  bool setPbWO2MaterialProperties_( G4Material* aMaterial );

  bool      useBirk,doCherenkov_;
  double    birk1, birk2, birk3;
  double    slopeLY;
  DimensionMap xtalLMap; // Store length and width

  /// Table of Cherenkov angle integrals vs photon momentum
  std::auto_ptr<G4PhysicsOrderedFreeVector> chAngleIntegrals_;

  // Histogramming
  TTree* ntuple_;
  int nphotons_;
  float px_[MAXPHOTONS],py_[MAXPHOTONS],pz_[MAXPHOTONS];
  float x_[MAXPHOTONS],y_[MAXPHOTONS],z_[MAXPHOTONS];

};

#endif // DreamSD_h
