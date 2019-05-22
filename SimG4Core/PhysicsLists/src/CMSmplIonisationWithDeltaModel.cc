//
// -------------------------------------------------------------------
//
//
// File name:     CMSmplIonisationWithDeltaModel
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 02.03.2019 copied from Geant4 10.5p01
//
//
// -------------------------------------------------------------------
// References
// [1] Steven P. Ahlen: Energy loss of relativistic heavy ionizing particles,
//     S.P. Ahlen, Rev. Mod. Phys 52(1980), p121
// [2] K.A. Milton arXiv:hep-ex/0602040
// [3] S.P. Ahlen and K. Kinoshita, Phys. Rev. D26 (1982) 2347

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "SimG4Core/PhysicsLists/interface/CMSmplIonisationWithDeltaModel.h"
#include "Randomize.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4Electron.hh"
#include "G4DynamicParticle.hh"
#include "G4ProductionCutsTable.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Log.hh"
#include "G4Pow.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

std::vector<G4double>* CMSmplIonisationWithDeltaModel::dedx0 = nullptr;

CMSmplIonisationWithDeltaModel::CMSmplIonisationWithDeltaModel(G4double mCharge, const G4String& nam)
    : G4VEmModel(nam),
      G4VEmFluctuationModel(nam),
      magCharge(mCharge),
      twoln10(std::log(100.0)),
      betalow(0.01),
      betalim(0.1),
      beta2lim(betalim * betalim),
      bg2lim(beta2lim * (1.0 + beta2lim)) {
  nmpl = G4lrint(std::abs(magCharge) * 2 * fine_structure_const);
  if (nmpl > 6) {
    nmpl = 6;
  } else if (nmpl < 1) {
    nmpl = 1;
  }
  pi_hbarc2_over_mc2 = pi * hbarc * hbarc / electron_mass_c2;
  chargeSquare = magCharge * magCharge;
  dedxlim = 45. * nmpl * nmpl * GeV * cm2 / g;
  fParticleChange = nullptr;
  theElectron = G4Electron::Electron();
  G4cout << "### Monopole ionisation model with d-electron production, Gmag= " << magCharge / eplus << G4endl;
  monopole = nullptr;
  mass = 0.0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

CMSmplIonisationWithDeltaModel::~CMSmplIonisationWithDeltaModel() {
  if (IsMaster()) {
    delete dedx0;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisationWithDeltaModel::SetParticle(const G4ParticleDefinition* p) {
  monopole = p;
  mass = monopole->GetPDGMass();
  G4double emin = std::min(LowEnergyLimit(), 0.1 * mass * (1. / sqrt(1. - betalow * betalow) - 1.));
  G4double emax = std::max(HighEnergyLimit(), 10 * mass * (1. / sqrt(1. - beta2lim) - 1.));
  SetLowEnergyLimit(emin);
  SetHighEnergyLimit(emax);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisationWithDeltaModel::Initialise(const G4ParticleDefinition* p, const G4DataVector&) {
  if (!monopole) {
    SetParticle(p);
  }
  if (!fParticleChange) {
    fParticleChange = GetParticleChangeForLoss();
  }
  if (IsMaster()) {
    if (!dedx0) {
      dedx0 = new std::vector<G4double>;
    }
    G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
    G4int numOfCouples = theCoupleTable->GetTableSize();
    G4int n = dedx0->size();
    if (n < numOfCouples) {
      dedx0->resize(numOfCouples);
    }
    G4Pow* g4calc = G4Pow::GetInstance();

    // initialise vector
    for (G4int i = 0; i < numOfCouples; ++i) {
      const G4Material* material = theCoupleTable->GetMaterialCutsCouple(i)->GetMaterial();
      G4double eDensity = material->GetElectronDensity();
      G4double vF = electron_Compton_length * g4calc->A13(3. * pi * pi * eDensity);
      (*dedx0)[i] = pi_hbarc2_over_mc2 * eDensity * nmpl * nmpl * (G4Log(2 * vF / fine_structure_const) - 0.5) / vF;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double CMSmplIonisationWithDeltaModel::MinEnergyCut(const G4ParticleDefinition*, const G4MaterialCutsCouple* couple) {
  return couple->GetMaterial()->GetIonisation()->GetMeanExcitationEnergy();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisationWithDeltaModel::ComputeDEDXPerVolume(const G4Material* material,
                                                              const G4ParticleDefinition* p,
                                                              G4double kineticEnergy,
                                                              G4double maxEnergy) {
  if (!monopole) {
    SetParticle(p);
  }
  G4double tmax = MaxSecondaryEnergy(p, kineticEnergy);
  G4double cutEnergy = std::min(tmax, maxEnergy);
  cutEnergy = std::max(LowEnergyLimit(), cutEnergy);
  G4double tau = kineticEnergy / mass;
  G4double gam = tau + 1.0;
  G4double bg2 = tau * (tau + 2.0);
  G4double beta2 = bg2 / (gam * gam);
  G4double beta = sqrt(beta2);

  // low-energy asymptotic formula
  G4double dedx = (*dedx0)[CurrentCouple()->GetIndex()] * beta;

  // above asymptotic
  if (beta > betalow) {
    // high energy
    if (beta >= betalim) {
      dedx = ComputeDEDXAhlen(material, bg2, cutEnergy);

    } else {
      G4double dedx1 = (*dedx0)[CurrentCouple()->GetIndex()] * betalow;
      G4double dedx2 = ComputeDEDXAhlen(material, bg2lim, cutEnergy);

      // extrapolation between two formula
      G4double kapa2 = beta - betalow;
      G4double kapa1 = betalim - beta;
      dedx = (kapa1 * dedx1 + kapa2 * dedx2) / (kapa1 + kapa2);
    }
  }
  return dedx;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisationWithDeltaModel::ComputeDEDXAhlen(const G4Material* material,
                                                          G4double bg2,
                                                          G4double cutEnergy) {
  G4double eDensity = material->GetElectronDensity();
  G4double eexc = material->GetIonisation()->GetMeanExcitationEnergy();

  // Ahlen's formula for nonconductors, [1]p157, f(5.7)
  G4double dedx = 0.5 * (G4Log(2.0 * electron_mass_c2 * bg2 * cutEnergy / (eexc * eexc)) - 1.0);

  // Kazama et al. cross-section correction
  G4double k = 0.406;
  if (nmpl > 1) {
    k = 0.346;
  }

  // Bloch correction
  const G4double B[7] = {0.0, 0.248, 0.672, 1.022, 1.243, 1.464, 1.685};

  dedx += 0.5 * k - B[nmpl];

  // density effect correction
  G4double x = G4Log(bg2) / twoln10;
  dedx -= material->GetIonisation()->DensityCorrection(x);

  // now compute the total ionization loss
  dedx *= pi_hbarc2_over_mc2 * eDensity * nmpl * nmpl;

  dedx = std::max(dedx, 0.0);
  return dedx;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double CMSmplIonisationWithDeltaModel::ComputeCrossSectionPerElectron(const G4ParticleDefinition* p,
                                                                        G4double kineticEnergy,
                                                                        G4double cut,
                                                                        G4double maxKinEnergy) {
  if (!monopole) {
    SetParticle(p);
  }
  G4double tmax = MaxSecondaryEnergy(p, kineticEnergy);
  G4double maxEnergy = std::min(tmax, maxKinEnergy);
  G4double cutEnergy = std::max(LowEnergyLimit(), cut);
  G4double cross =
      (cutEnergy < maxEnergy) ? (0.5 / cutEnergy - 0.5 / maxEnergy) * pi_hbarc2_over_mc2 * nmpl * nmpl : 0.0;
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double CMSmplIonisationWithDeltaModel::ComputeCrossSectionPerAtom(const G4ParticleDefinition* p,
                                                                    G4double kineticEnergy,
                                                                    G4double Z,
                                                                    G4double,
                                                                    G4double cutEnergy,
                                                                    G4double maxEnergy) {
  G4double cross = Z * ComputeCrossSectionPerElectron(p, kineticEnergy, cutEnergy, maxEnergy);
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisationWithDeltaModel::SampleSecondaries(vector<G4DynamicParticle*>* vdp,
                                                       const G4MaterialCutsCouple*,
                                                       const G4DynamicParticle* dp,
                                                       G4double minKinEnergy,
                                                       G4double maxEnergy) {
  G4double kineticEnergy = dp->GetKineticEnergy();
  G4double tmax = MaxSecondaryEnergy(dp->GetDefinition(), kineticEnergy);

  G4double maxKinEnergy = std::min(maxEnergy, tmax);
  if (minKinEnergy >= maxKinEnergy) {
    return;
  }

  //G4cout << "CMSmplIonisationWithDeltaModel::SampleSecondaries: E(GeV)= "
  //         << kineticEnergy/GeV << " M(GeV)= " << mass/GeV
  //         << " tmin(MeV)= " << minKinEnergy/MeV << G4endl;

  G4double totEnergy = kineticEnergy + mass;
  G4double etot2 = totEnergy * totEnergy;
  G4double beta2 = kineticEnergy * (kineticEnergy + 2.0 * mass) / etot2;

  // sampling without nuclear size effect
  G4double q = G4UniformRand();
  G4double deltaKinEnergy = minKinEnergy * maxKinEnergy / (minKinEnergy * (1.0 - q) + maxKinEnergy * q);

  // delta-electron is produced
  G4double totMomentum = totEnergy * sqrt(beta2);
  G4double deltaMomentum = sqrt(deltaKinEnergy * (deltaKinEnergy + 2.0 * electron_mass_c2));
  G4double cost = deltaKinEnergy * (totEnergy + electron_mass_c2) / (deltaMomentum * totMomentum);
  cost = std::min(cost, 1.0);

  G4double sint = sqrt((1.0 - cost) * (1.0 + cost));

  G4double phi = twopi * G4UniformRand();

  G4ThreeVector deltaDirection(sint * cos(phi), sint * sin(phi), cost);
  const G4ThreeVector& direction = dp->GetMomentumDirection();
  deltaDirection.rotateUz(direction);

  // create G4DynamicParticle object for delta ray
  G4DynamicParticle* delta = new G4DynamicParticle(theElectron, deltaDirection, deltaKinEnergy);

  vdp->push_back(delta);

  // Change kinematics of primary particle
  kineticEnergy -= deltaKinEnergy;
  G4ThreeVector finalP = direction * totMomentum - deltaDirection * deltaMomentum;
  finalP = finalP.unit();

  fParticleChange->SetProposedKineticEnergy(kineticEnergy);
  fParticleChange->SetProposedMomentumDirection(finalP);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisationWithDeltaModel::SampleFluctuations(const G4MaterialCutsCouple* couple,
                                                            const G4DynamicParticle* dp,
                                                            G4double tmax,
                                                            G4double length,
                                                            G4double meanLoss) {
  G4double siga = Dispersion(couple->GetMaterial(), dp, tmax, length);
  G4double loss = meanLoss;
  siga = sqrt(siga);
  G4double twomeanLoss = meanLoss + meanLoss;

  if (twomeanLoss < siga) {
    G4double x;
    do {
      loss = twomeanLoss * G4UniformRand();
      x = (loss - meanLoss) / siga;
      // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    } while (1.0 - 0.5 * x * x < G4UniformRand());
  } else {
    do {
      loss = G4RandGauss::shoot(meanLoss, siga);
      // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    } while (0.0 > loss || loss > twomeanLoss);
  }
  return loss;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisationWithDeltaModel::Dispersion(const G4Material* material,
                                                    const G4DynamicParticle* dp,
                                                    G4double tmax,
                                                    G4double length) {
  G4double siga = 0.0;
  G4double tau = dp->GetKineticEnergy() / mass;
  if (tau > 0.0) {
    G4double electronDensity = material->GetElectronDensity();
    G4double gam = tau + 1.0;
    G4double invbeta2 = (gam * gam) / (tau * (tau + 2.0));
    siga = (invbeta2 - 0.5) * twopi_mc2_rcl2 * tmax * length * electronDensity * chargeSquare;
  }
  return siga;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisationWithDeltaModel::MaxSecondaryEnergy(const G4ParticleDefinition*, G4double kinEnergy) {
  G4double tau = kinEnergy / mass;
  return 2.0 * electron_mass_c2 * tau * (tau + 2.);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
