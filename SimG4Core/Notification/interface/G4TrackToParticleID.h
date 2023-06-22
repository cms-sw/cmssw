#ifndef SimG4Core_G4TrackToParticleID_H
#define SimG4Core_G4TrackToParticleID_H

class G4Track;
class G4PrimaryParticle;

class G4TrackToParticleID {
public:
  // CMS convention (different from ordinary PDG code)
  static int particleID(const G4Track *);
  static int particleID(const G4PrimaryParticle *, const int id);

  static bool isGammaElectronPositron(int pdgCode);
  static bool isGammaElectronPositron(const G4Track *);

  static bool isMuon(int pdgCode);
  static bool isMuon(const G4Track *);

  // pi+-, p, pbar, n, nbar, KL, K+-, light ion and anti-ion, generic ion
  static bool isStableHadron(int pdgCode);

  // pi+-, p, pbar, n, nbar, KL, K+-, light ions and anti-ions
  static bool isStableHadronIon(const G4Track *);
};

#endif
