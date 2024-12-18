
#include "G4NucleiProperties.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "TMath.h"
#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/CustomPhysics/interface/CMSSQNeutronAnnih.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQ.h"

CMSSQNeutronAnnih::CMSSQNeutronAnnih(double mass) : G4HadronicInteraction("SexaQuark-neutron annihilation") {
  SetMinEnergy(0.0 * GeV);
  SetMaxEnergy(100. * TeV);

  theSQ = CMSSQ::SQ(mass);
  theK0S = G4KaonZeroShort::KaonZeroShort();
  theAntiL = G4AntiLambda::AntiLambda();
  theProton = G4Proton::
      Proton();  //proton only used when the particle which the sexaquark hits is a deutereon and the neutron dissapears, so what stays behind is a proton
}

CMSSQNeutronAnnih::~CMSSQNeutronAnnih() {}

//9Be momentum distribution from Jan Ryckebusch
G4double CMSSQNeutronAnnih::momDistr(G4double x_in) {
  const int n_entries = 50;

  G4double CDF_k[n_entries] = {0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,   1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                               1.7, 1.8, 1.9, 2,   2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,   3.1, 3.2, 3.3,
                               3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.,  4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9};

  G4double x[n_entries] = {0,
                           0.0038033182,
                           0.0187291764,
                           0.0510409777,
                           0.1048223609,
                           0.1807862863,
                           0.2756514534,
                           0.3825832103,
                           0.4926859745,
                           0.5970673837,
                           0.6887542272,
                           0.7637748784,
                           0.8212490273,
                           0.8627259608,
                           0.8911605331,
                           0.9099115186,
                           0.9220525854,
                           0.9300190818,
                           0.9355376091,
                           0.9397242185,
                           0.9432387722,
                           0.946438928,
                           0.9495023924,
                           0.9525032995,
                           0.9554669848,
                           0.9583936672,
                           0.9612770117,
                           0.9641067202,
                           0.9668727859,
                           0.9695676121,
                           0.9721815799,
                           0.9747092981,
                           0.9771426396,
                           0.9794740235,
                           0.9816956807,
                           0.9838003583,
                           0.9857816165,
                           0.9876331761,
                           0.9893513365,
                           0.9909333198,
                           0.992378513,
                           0.9936885054,
                           0.9948665964,
                           0.9959179448,
                           0.9968491104,
                           0.9976680755,
                           0.9983832508,
                           0.9990041784,
                           0.9995400073,
                           1};

  //now interpolate the above points for x_in
  if (x_in <= 0.0)
    return 0.;
  if (x_in >= 1.0)
    return CDF_k[n_entries - 1];
  for (int i = 1; i < n_entries; i++) {
    if (x[i] >= x_in) {
      return (CDF_k[i] - CDF_k[i - 1]) / (x[i] - x[i - 1]) * (x_in - x[i - 1]) + CDF_k[i - 1];
    }
  }
  return 0.0;
}

G4HadFinalState* CMSSQNeutronAnnih::ApplyYourself(const G4HadProjectile& aTrack, G4Nucleus& targetNucleus) {
  theParticleChange.Clear();
  const G4HadProjectile* aParticle = &aTrack;
  G4double ekin = aParticle->GetKineticEnergy();

  G4int A = targetNucleus.GetA_asInt();
  G4int Z = targetNucleus.GetZ_asInt();

  G4double m_K0S = G4KaonZeroShort::KaonZeroShort()->GetPDGMass();
  G4double m_L = G4AntiLambda::AntiLambda()->GetPDGMass();

  //G4double plab = aParticle->GetTotalMomentum();

  //    edm::LogVerbatim("CMSSWNeutronAnnih") << "CMSSQNeutronAnnih: Incident particle p (GeV), total Energy (GeV), particle name, eta ="
  //       << plab/GeV << "  "
  //       << aParticle->GetTotalEnergy()/GeV << "  "
  //       << aParticle->GetDefinition()->GetParticleName() << " "
  //	   << aParticle->Get4Momentum();

  // Scattered particle referred to axis of incident particle
  //const G4ParticleDefinition* theParticle = aParticle->GetDefinition();

  //G4int projPDG = theParticle->GetPDGEncoding();
  //    edm::LogVerbatim("CMSSWNeutronAnnih") << "CMSSQNeutronAnnih: for " << theParticle->GetParticleName()
  //           << " PDGcode= " << projPDG << " on nucleus Z= " << Z
  //           << " A= " << A << " N= " << N;

  const G4LorentzVector& lv1 = aParticle->Get4Momentum();
  edm::LogVerbatim("CMSSWNeutronAnnih") << "The neutron Fermi momentum (mag, x, y, z) "
                                        << targetNucleus.GetFermiMomentum().mag() / MeV << " "
                                        << targetNucleus.GetFermiMomentum().x() / MeV << " "
                                        << targetNucleus.GetFermiMomentum().y() / MeV << " "
                                        << targetNucleus.GetFermiMomentum().z() / MeV;

  //calculate fermi momentum

  G4double k_neutron = momDistr(G4UniformRand());
  G4double momentum_neutron = 0.1973 * GeV * k_neutron;

  G4double theta_neutron = TMath::ACos(2 * G4UniformRand() - 1);
  G4double phi_neutron = 2. * TMath::Pi() * G4UniformRand();

  G4double p_neutron_x = momentum_neutron * TMath::Sin(theta_neutron) * TMath::Cos(phi_neutron);
  G4double p_neutron_y = momentum_neutron * TMath::Sin(theta_neutron) * TMath::Sin(phi_neutron);
  G4double p_neutron_z = momentum_neutron * TMath::Cos(theta_neutron);

  //G4LorentzVector lv0(targetNucleus.GetFermiMomentum(), sqrt( pow(G4Neutron::Neutron()->GetPDGMass(),2) + targetNucleus.GetFermiMomentum().mag2()  ) );
  G4LorentzVector lv0(p_neutron_x,
                      p_neutron_y,
                      p_neutron_z,
                      sqrt(pow(G4Neutron::Neutron()->GetPDGMass(), 2) + momentum_neutron * momentum_neutron));

  //const G4Nucleus* aNucleus = &targetNucleus;
  G4double BENeutronInNucleus = 0;
  if (A != 0)
    BENeutronInNucleus = G4NucleiProperties::GetBindingEnergy(A, Z) / (A);

  edm::LogVerbatim("CMSSWNeutronAnnih") << "BE of nucleon in the nucleus (GeV): " << BENeutronInNucleus / GeV;

  G4LorentzVector lvBE(0, 0, 0, BENeutronInNucleus / GeV);
  G4LorentzVector lv = lv0 + lv1 - lvBE;

  // kinematiacally impossible ?
  G4double etot = lv0.e() + lv1.e() - lvBE.e();
  if (etot < theK0S->GetPDGMass() + theAntiL->GetPDGMass()) {
    theParticleChange.SetEnergyChange(ekin);
    theParticleChange.SetMomentumChange(aTrack.Get4Momentum().vect().unit());
    return &theParticleChange;
  }

  float newIonMass = targetNucleus.AtomicMass(A - 1, Z) * 931.5 * MeV;
  ;
  G4LorentzVector nlvIon(0, 0, 0, newIonMass);

  G4double theta_KS0_star = TMath::ACos(2 * G4UniformRand() - 1);
  G4double phi_KS0_star = 2. * TMath::Pi() * G4UniformRand();

  G4double p_K0S_star_x = TMath::Sin(theta_KS0_star) * TMath::Cos(phi_KS0_star);
  G4double p_K0S_star_y = TMath::Sin(theta_KS0_star) * TMath::Sin(phi_KS0_star);
  G4double p_K0S_star_z = TMath::Cos(theta_KS0_star);

  G4ThreeVector p(p_K0S_star_x, p_K0S_star_y, p_K0S_star_z);
  double m0 = lv.m();
  double m0_2 = m0 * m0;
  double m1_2 = m_K0S * m_K0S;
  double m2_2 = m_L * m_L;

  p *= 0.5 / m0 * sqrt(m0_2 * m0_2 + m1_2 * m1_2 + m2_2 * m2_2 - 2 * m0_2 * m1_2 - 2 * m0_2 * m2_2 - 2 * m1_2 * m2_2);
  double p2 = p.mag2();

  G4LorentzVector nlvK0S(p, sqrt(p2 + m1_2));
  G4LorentzVector nlvAntiL(-p, sqrt(p2 + m2_2));

  // Boost out of the rest frame.
  nlvK0S.boost(lv.boostVector());
  nlvAntiL.boost(lv.boostVector());

  // now move to implement the interaction
  theParticleChange.SetStatusChange(stopAndKill);
  //theParticleChange.SetEnergyChange(ekin); // was 0.0

  G4DynamicParticle* aSec1 = new G4DynamicParticle(theK0S, nlvK0S);
  theParticleChange.AddSecondary(aSec1);
  G4DynamicParticle* aSec2 = new G4DynamicParticle(theAntiL, nlvAntiL);
  theParticleChange.AddSecondary(aSec2);

  const G4ParticleDefinition* theRemainingNucleusDef = theProton;
  if (A != 1)
    theRemainingNucleusDef = G4IonTable::GetIonTable()->GetIon(Z, A - 1);
  G4DynamicParticle* aSec3 = new G4DynamicParticle(theRemainingNucleusDef, nlvIon);
  theParticleChange.AddSecondary(aSec3);

  // return as is; we don't care about what happens to the nucleus
  return &theParticleChange;
}
