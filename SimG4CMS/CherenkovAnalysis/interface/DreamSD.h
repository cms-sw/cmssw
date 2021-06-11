#ifndef SimG4CMS_DreamSD_h
#define SimG4CMS_DreamSD_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "G4PhysicsFreeVector.hh"

#include <DD4hep/DD4hepUnits.h>

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
  void fillMap(const std::string &, double, double);
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

  static constexpr double k_ScaleFromDDDToG4 = 1.0;
  static constexpr double k_ScaleFromDD4HepToG4 = 1.0 / dd4hep::mm;

  bool useBirk_, doCherenkov_, readBothSide_, dd4hep_;
  double birk1_, birk2_, birk3_;
  double slopeLY_;
  DimensionMap xtalLMap_;  // Store length and width

  int side_;

  /// Table of Cherenkov angle integrals vs photon momentum
  std::unique_ptr<G4PhysicsFreeVector> chAngleIntegrals_;
  G4MaterialPropertiesTable *materialPropertiesTable_;

  int nphotons_;
};

#endif  // DreamSD_h
