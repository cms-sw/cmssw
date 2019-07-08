#ifndef SimG4Core_PhysicsLists_CMSTrackingCutModel_H
#define SimG4Core_PhysicsLists_CMSTrackingCutModel_H
//
// Vladimir Ivanchenko 27.06.2019
//
// This is the base model of low-energy parameterisation
// applicable for e+- in Ecal and passive absorbers
//

#include "globals.hh"

class G4ParticleDefinition;

class CMSTrackingCutModel {
public:
  explicit CMSTrackingCutModel(const G4ParticleDefinition *);
  virtual ~CMSTrackingCutModel();

  virtual G4double SampleEnergyDepositEcal(G4double kinEnergy);

  inline void InitialiseForStep(G4double fac, G4double rms);

protected:
  const G4ParticleDefinition *particle_;

  G4double deltaE_;
  G4double factor_;
  G4double rms_;
};

inline void CMSTrackingCutModel::InitialiseForStep(G4double fac, G4double rms) {
  factor_ = fac;
  rms_ = rms;
}

#endif
