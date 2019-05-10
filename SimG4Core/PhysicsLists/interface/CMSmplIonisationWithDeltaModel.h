//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
//
//
// File name:     CMSmplIonisationWithDeltaModel
//
// Author:        Vladimir Ivanchenko copied from Geant4 10.5p01
//
// Creation date: 02.03.2019
//
// Class Description:
//
// Implementation of model of energy loss of the magnetic monopole

// -------------------------------------------------------------------
//

#ifndef CMSmplIonisationWithDeltaModel_h
#define CMSmplIonisationWithDeltaModel_h 1

#include "G4VEmModel.hh"
#include "G4VEmFluctuationModel.hh"
#include <vector>

class G4ParticleChangeForLoss;

class CMSmplIonisationWithDeltaModel : public G4VEmModel, public G4VEmFluctuationModel {
public:
  explicit CMSmplIonisationWithDeltaModel(G4double mCharge, const G4String& nam = "mplIonisationWithDelta");

  ~CMSmplIonisationWithDeltaModel() override;

  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;

  G4double ComputeDEDXPerVolume(const G4Material*,
                                const G4ParticleDefinition*,
                                G4double kineticEnergy,
                                G4double cutEnergy) override;

  virtual G4double ComputeCrossSectionPerElectron(const G4ParticleDefinition*,
                                                  G4double kineticEnergy,
                                                  G4double cutEnergy,
                                                  G4double maxEnergy);

  G4double ComputeCrossSectionPerAtom(const G4ParticleDefinition*,
                                      G4double kineticEnergy,
                                      G4double Z,
                                      G4double A,
                                      G4double cutEnergy,
                                      G4double maxEnergy) override;

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                         const G4MaterialCutsCouple*,
                         const G4DynamicParticle*,
                         G4double tmin,
                         G4double maxEnergy) override;

  G4double SampleFluctuations(const G4MaterialCutsCouple*,
                              const G4DynamicParticle*,
                              G4double tmax,
                              G4double length,
                              G4double meanLoss) override;

  G4double Dispersion(const G4Material*, const G4DynamicParticle*, G4double tmax, G4double length) override;

  G4double MinEnergyCut(const G4ParticleDefinition*, const G4MaterialCutsCouple* couple) override;

  void SetParticle(const G4ParticleDefinition* p);

protected:
  G4double MaxSecondaryEnergy(const G4ParticleDefinition*, G4double kinEnergy) override;

private:
  G4double ComputeDEDXAhlen(const G4Material* material, G4double bg2, G4double cut);

  const G4ParticleDefinition* monopole;
  G4ParticleDefinition* theElectron;
  G4ParticleChangeForLoss* fParticleChange;

  G4double mass;
  G4double magCharge;
  G4double twoln10;
  G4double betalow;
  G4double betalim;
  G4double beta2lim;
  G4double bg2lim;
  G4double chargeSquare;
  G4double dedxlim;
  G4int nmpl;
  G4double pi_hbarc2_over_mc2;

  static std::vector<G4double>* dedx0;
};

#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
