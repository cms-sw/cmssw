#ifndef SimG4CMS_DreamSD_h
#define SimG4CMS_DreamSD_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4PhysicsOrderedFreeVector.hh"

#include <map>

const int MAXPHOTONS = 500;  // Maximum number of photons we can store

class G4LogicalVolume;

class DreamSD : public CaloSD {
public:
  DreamSD(const std::string &,
          const edm::EventSetup &,
          const SensitiveDetectorCatalog &,
          edm::ParameterSet const &,
          const SimTrackManager *);
  ~DreamSD() override {}

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  double getEnergyDeposit(const G4Step *) override;
  void initRun() override;

private:
  typedef std::pair<double, double> Doubles;
  typedef std::map<G4LogicalVolume *, Doubles> DimensionMap;

  void initMap(const std::string &, const edm::EventSetup &);
  double curve_LY(const G4Step *, int);
  double crystalLength(G4LogicalVolume *) const;
  double crystalWidth(G4LogicalVolume *) const;

  /// Returns the total energy due to Cherenkov radiation
  double cherenkovDeposit_(const G4Step *aStep);
  /// Returns average number of photons created by track
  double getAverageNumberOfPhotons_(const double charge,
                                    const double beta,
                                    const G4Material *aMaterial,
                                    const G4MaterialPropertyVector *rIndex);
  /// Returns energy deposit for a given photon
  double getPhotonEnergyDeposit_(const G4ParticleMomentum &p, const G4ThreeVector &x, const G4Step *aStep);
  /// Sets material properties at run-time...
  bool setPbWO2MaterialProperties_(G4Material *aMaterial);

  bool useBirk, doCherenkov_, readBothSide_;
  double birk1, birk2, birk3;
  double slopeLY;
  DimensionMap xtalLMap;  // Store length and width

  int side;

  /// Table of Cherenkov angle integrals vs photon momentum
  std::unique_ptr<G4PhysicsOrderedFreeVector> chAngleIntegrals_;
  G4MaterialPropertiesTable *materialPropertiesTable;

  int nphotons_;
};

#endif  // DreamSD_h
