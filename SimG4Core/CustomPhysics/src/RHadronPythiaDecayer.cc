#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayer.h"
#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayDataManager.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4Step.hh"
#include "G4DecayProducts.hh"
#include "G4VParticleChange.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4TransportationManager.hh"

#include "Pythia8/Pythia.h"
#include "Pythia8/RHadrons.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>

RHadronPythiaDecayDataManager* gRHadronPythiaDecayDataManager = new RHadronPythiaDecayDataManager();
static inline unsigned short int nth_digit(const int& val, const unsigned short& n) {
  return (std::abs(val) / (int(std::pow(10, n - 1)))) % 10;
}

RHadronPythiaDecayer::RHadronPythiaDecayer(edm::ParameterSet const& p) : pythia_(new Pythia8::Pythia()) {
  std::string SLHAParticleDefinitionsFile = p.getParameter<edm::FileInPath>("particlesDef").fullPath();
  std::string commandFile = p.getParameter<edm::FileInPath>("RhadronPythiaDecayerCommandFile").fullPath();

  // Read in the SLHA particle definitions file
  edm::LogVerbatim("SimG4CoreCustomPhysics")
      << "RHadronPythiaDecayer: Using SLHA particle definitions file: " << SLHAParticleDefinitionsFile;
  pythia_->readString("SLHA:file = " + SLHAParticleDefinitionsFile);

  // Read in the command file for Pythia8 settings
  edm::LogVerbatim("SimG4CoreCustomPhysics") << "RHadronPythiaDecayer: Using command file: " << commandFile;
  std::string line;
  std::ifstream command_stream(commandFile);
  while (getline(command_stream, line)) {
    pythia_->readString(line);
    edm::LogVerbatim("SimG4CoreCustomPhysics") << "RHadronPythiaDecayer: Pythia8 command: " << line;
  }
  command_stream.close();
  pythia_->init();
}

RHadronPythiaDecayer::~RHadronPythiaDecayer() {
  if (GetExtDecayer() == this)
    SetExtDecayer(nullptr);
}

G4VParticleChange* RHadronPythiaDecayer::DecayIt(const G4Track& aTrack, const G4Step& aStep) {
  // First, clear the secondary displacements and call the standard DecayIt to generate secondaries
  secondaryDisplacements_.clear();
  gRHadronPythiaDecayDataManager->addDecayParent(aTrack);
  G4VParticleChange* fParticleChangeForDecay = G4Decay::DecayIt(aTrack, aStep);

  // Update the position of the secondaries in geant to match the potentially displaced positions from pythia. The list is stored in reverse order
  G4int secondaryDisplacementIndex = 0;
  for (G4int i = fParticleChangeForDecay->GetNumberOfSecondaries() - 1; i >= 0; --i) {
    G4Track* secondary = fParticleChangeForDecay->GetSecondary(i);
    secondary->SetPosition(secondary->GetPosition() + secondaryDisplacements_[secondaryDisplacementIndex]);
    gRHadronPythiaDecayDataManager->addDecayDaughter(*secondary);
    ++secondaryDisplacementIndex;
  }

  return fParticleChangeForDecay;
}

G4DecayProducts* RHadronPythiaDecayer::ImportDecayProducts(const G4Track& aTrack) {
  // Initialize decay products. These will be used inside of G4Decay::DecayIt()
  G4DecayProducts* dp = new G4DecayProducts();
  dp->SetParentParticle(*(aTrack.GetDynamicParticle()));
  std::vector<G4DynamicParticle*> particles;

  // Use Pythia8 to decay the particle and add them to the particles vector
  pythiaDecay(aTrack, particles);

  // Add the particles to the decay products
  for (unsigned int i = 0; i < particles.size(); ++i) {
    if (particles[i])
      dp->PushProducts(particles[i]);
  }

  return dp;
}

void RHadronPythiaDecayer::pythiaDecay(const G4Track& aTrack, std::vector<G4DynamicParticle*>& particles) {
  // Initialize the Pythia8 event where the decay will happen
  Pythia8::Event& event = pythia_->event;

  // Store the decay location and world volume to later check if decay products are inside the world volume
  const G4ThreeVector& RHadronDecayLocation = aTrack.GetPosition();
  G4VPhysicalVolume* worldPhys =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  G4VSolid* worldSolid = worldPhys->GetLogicalVolume()->GetSolid();

  // Fill the event with the Rhadron, strip it down to its constituents, i.e. gluino and quarks for a gluino R-hadron. Then finally let pythia handle the rest
  fillParticle(aTrack, event);
  RHadronToConstituents(event);
  pythia_->next();

  // Add the particles from the Pythia event into the Geant particle vector
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  for (int i = 0; i < pythia_->event.size(); i++) {
    // If the particle status is negative and it decays outside of the vertex, change the status to positive and do not add its decay products. If its status is negative but it decays inside the detector, skip it.
    G4ThreeVector vProd =
        RHadronDecayLocation +
        G4ThreeVector(pythia_->event[i].xProd(), pythia_->event[i].yProd(), pythia_->event[i].zProd());
    G4ThreeVector vDec = RHadronDecayLocation +
                         G4ThreeVector(pythia_->event[i].xDec(), pythia_->event[i].yDec(), pythia_->event[i].zDec());
    if (pythia_->event[i].status() < 0 && (worldSolid->Inside(vDec) == kOutside))
      pythia_->event[i].statusPos();
    else if (pythia_->event[i].status() < 0)
      continue;
    if (worldSolid->Inside(vProd) == kOutside)
      continue;

    G4ThreeVector displacement(pythia_->event[i].xProd(), pythia_->event[i].yProd(), pythia_->event[i].zProd());
    G4LorentzVector p4(pythia_->event[i].px(), pythia_->event[i].py(), pythia_->event[i].pz(), pythia_->event[i].e());
    p4 *= CLHEP::GeV;

    // Get the particle definition from the Pythia event
    const G4ParticleDefinition* particleDefinition = particleTable->FindParticle(pythia_->event[i].id());
    if (!particleDefinition) {
      edm::LogWarning("SimG4CoreCustomPhysics") << "RHadronPythiaDecayer: I don't know a definition for pdgid "
                                                << pythia_->event[i].id() << "! Skipping it...";
      continue;
    }

    // Create the dynamic particle and add it to Geant
    G4DynamicParticle* dynamicParticle = new G4DynamicParticle(particleDefinition, p4);
    particles.push_back(dynamicParticle);
    // Store the position of the secondary particle to update in RHadronPythiaDecayer::DecayIt
    secondaryDisplacements_.push_back(displacement);
  }
}

void RHadronPythiaDecayer::fillParticle(const G4Track& aTrack, Pythia8::Event& event) const {
  // Reset event record to allow for new event.
  event.reset();

  // Get particle mass and 4-momentum.
  double mass = aTrack.GetDynamicParticle()->GetMass() / CLHEP::GeV;
  const G4LorentzVector g4p4 = aTrack.GetDynamicParticle()->Get4Momentum() / CLHEP::GeV;
  Pythia8::Vec4 p4(g4p4.px(), g4p4.py(), g4p4.pz(), g4p4.e());

  // Store the particle in the event record.
  event.append(aTrack.GetDefinition()->GetPDGEncoding(), 1, 0, 0, p4, mass);
}

void RHadronPythiaDecayer::RHadronToConstituents(Pythia8::Event& event) {
  // This code is very similar to Pythia8::RHadrons::decay(). Unfortunately, it is not possible in this scenario to use Pythia8::RHadrons::decay().
  // Because we need to use a new instance of pythia, the value of nRHad inside of Pythia8::RHadrons is set to 0 and the for loop inside of Pythia8::RHadrons::decay() never runs.
  // As far as I'm aware, it is impossible to update nRHad without first producing R-hadrons with the pythia instance, which is not what we want to do.
  // So, in lieu of this, code from Pythia8::RHadrons::decay() has been pasted here rather than used directly.

  Pythia8::ParticleData& pdt = pythia_->particleData;

  int iRNow = 1;
  int idRHad = event[iRNow].id();
  double mRHad = event[iRNow].m();
  int iR0 = 0;
  int iR2 = 0;

  bool isTriplet = !isGluinoRHadron(idRHad);

  // Find flavour content of squark or gluino R-hadron.
  std::pair<int, int> idPair = (isTriplet) ? fromIdWithSquark(idRHad) : fromIdWithGluino(idRHad, &(pythia_->rndm));
  int id1 = idPair.first;
  int id2 = idPair.second;

  // Sharing of momentum: the squark/gluino should be restored
  // to original mass, but error if negative-mass spectators.
  int idRSb = pythia_->settings.mode("RHadrons:idSbottom");
  int idRSt = pythia_->settings.mode("RHadrons:idStop");
  int idRGo = pythia_->settings.mode("RHadrons:idGluino");
  int idLight = (abs(idRHad) - 1000000) / 10;
  int idSq = (idLight < 100) ? idLight / 10 : idLight / 100;
  int idRSq = (idSq == 6) ? idRSt : idRSb;

  // Handling R-Hadrons with anti-squarks
  idRSq = idRSq * std::copysign(1, idRHad);

  int idRBef = isTriplet ? idRSq : idRGo;

  // Mass of the underlying sparticle
  double mRBef = pdt.mSel(idRBef);

  // Fraction of the RHadron mass given by the sparticle
  double fracR = mRBef / mRHad;
  if (fracR >= 1.) {
    edm::LogError("SimG4CoreCustomPhysics")
        << "RHadronPythiaDecayer::RhadronToConstituents: R-hadron mass too low for decay.";
    return;
  }

  // Squark case
  if (isTriplet) {
    const int col =
        (pdt.colType(idRBef) != 0)
            ? event.nextColTag()
            : 0;  // NB There should be no way that this can be zero (see discussion on ATLASSIM-6687), but leaving check in there just in case something changes in the future.
    int tmpSparticleColor = id1 > 0 ? col : 0;
    int tmpSparticleAnticolor = id1 > 0 ? 0 : col;

    // Store the constituents of a squark R-hadron.

    // Sparticle
    // (id, status, mother1, mother2, daughter1, daughter2, col, acol, px, py, pz, e, m=0., scaleIn=0., polIn=9.)
    iR0 = event.append(
        id1, 106, iRNow, 0, 0, 0, tmpSparticleColor, tmpSparticleAnticolor, fracR * event[iRNow].p(), fracR * mRHad, 0.);
    // Spectator quark
    iR2 = event.append(id2,
                       106,
                       iRNow,
                       0,
                       0,
                       0,
                       tmpSparticleAnticolor,
                       tmpSparticleColor,
                       (1. - fracR) * event[iRNow].p(),
                       (1. - fracR) * mRHad,
                       0.);
  }
  // Gluino case
  else {
    double mOffsetCloudRH = 0.2;  // could be read from internal data?
    double m1Eff = pdt.constituentMass(id1) + mOffsetCloudRH;
    double m2Eff = pdt.constituentMass(id2) + mOffsetCloudRH;
    double frac1 = (1. - fracR) * m1Eff / (m1Eff + m2Eff);
    double frac2 = (1. - fracR) * m2Eff / (m1Eff + m2Eff);

    // Two new colours needed in the breakups.
    int col1 = event.nextColTag();
    int col2 = event.nextColTag();

    // Store the constituents of a gluino R-hadron.
    iR0 = event.append(idRBef, 106, iRNow, 0, 0, 0, col2, col1, fracR * event[iRNow].p(), fracR * mRHad, 0.);
    event.append(id1, 106, iRNow, 0, 0, 0, col1, 0, frac1 * event[iRNow].p(), frac1 * mRHad, 0.);
    iR2 = event.append(id2, 106, iRNow, 0, 0, 0, 0, col2, frac2 * event[iRNow].p(), frac2 * mRHad, 0.);
  }

  // Mark R-hadron as decayed and update history.
  event[iRNow].statusNeg();
  event[iRNow].daughters(iR0, iR2);

  // Set secondary vertex for decay products, but no lifetime.
  Pythia8::Vec4 vDec = event[iRNow].vProd() + event[iRNow].tau() * event[iR0].p() / event[iR0].m();
  for (int iRd = iR0; iRd <= iR2; ++iRd) {
    event[iRd].vProd(vDec);
  }
}

std::pair<int, int> RHadronPythiaDecayer::fromIdWithSquark(int idRHad) const {
  // Find squark flavour content.
  int idRSb = pythia_->settings.mode("RHadrons:idSbottom");
  int idRSt = pythia_->settings.mode("RHadrons:idStop");
  int idLight = (abs(idRHad) - 1000000) / 10;
  int idSq = (idLight < 100) ? idLight / 10 : idLight / 100;
  int id1 = (idSq == 6) ? idRSt : idRSb;
  if (idRHad < 0)
    id1 = -id1;

  // Find light (di)quark flavour content.
  int id2 = (idLight < 100) ? idLight % 10 : idLight % 100;
  if (id2 > 10)
    id2 = 100 * id2 + abs(idRHad) % 10;
  if ((id2 < 10 && idRHad > 0) || (id2 > 10 && idRHad < 0))
    id2 = -id2;

  return std::make_pair(id1, id2);
}

std::pair<int, int> RHadronPythiaDecayer::fromIdWithGluino(int idRHad, Pythia8::Rndm* rndmPtr) const {
  // Find light flavour content of R-hadron.
  int idLight = (abs(idRHad) - 1000000) / 10;
  int id1, id2, idTmp, idA, idB, idC;
  double diquarkSpin1RH = 0.5;

  // Gluinoballs: split g into d dbar or u ubar.
  if (idLight < 100) {
    id1 = (rndmPtr->flat() < 0.5) ? 1 : 2;
    id2 = -id1;

    // Gluino-meson: split into q + qbar.
  } else if (idLight < 1000) {
    id1 = (idLight / 10) % 10;
    id2 = -(idLight % 10);
    // Flip signs when first quark of down-type.
    if (id1 % 2 == 1) {
      idTmp = id1;
      id1 = -id2;
      id2 = -idTmp;
    }

    // Gluino-baryon: split to q + qq (diquark).
    // Pick diquark at random, except if c or b involved.
  } else {
    idA = (idLight / 100) % 10;
    idB = (idLight / 10) % 10;
    idC = idLight % 10;
    double rndmQ = 3. * rndmPtr->flat();
    if (idA > 3)
      rndmQ = 0.5;
    if (rndmQ < 1.) {
      id1 = idA;
      id2 = 1000 * idB + 100 * idC + 3;
      if (idB != idC && rndmPtr->flat() > diquarkSpin1RH)
        id2 -= 2;
    } else if (rndmQ < 2.) {
      id1 = idB;
      id2 = 1000 * idA + 100 * idC + 3;
      if (idA != idC && rndmPtr->flat() > diquarkSpin1RH)
        id2 -= 2;
    } else {
      id1 = idC;
      id2 = 1000 * idA + 100 * idB + 3;
      if (idA != idB && rndmPtr->flat() > diquarkSpin1RH)
        id2 -= 2;
    }
  }
  // Flip signs for anti-R-hadron.
  if (idRHad < 0) {
    idTmp = id1;
    id1 = -id2;
    id2 = -idTmp;
  }

  return std::make_pair(id1, id2);
}

bool RHadronPythiaDecayer::isGluinoRHadron(int pdgId) const {
  // Checking what kind of RHadron this is based on the digits in its PDGID
  const unsigned short digitValue_q1 = nth_digit(pdgId, 4);
  const unsigned short digitValue_l = nth_digit(pdgId, 5);

  // Gluino R-Hadrons have the form 109xxxx or 1009xxx
  if (digitValue_l == 9 || (digitValue_l == 0 && digitValue_q1 == 9)) {
    // This is a gluino R-Hadron
    return true;
  }

  // Special case : R-gluinoball
  if (pdgId == 1000993)
    return true;

  // This is not a gluino R-Hadron (probably a squark R-Hadron)
  return false;
}