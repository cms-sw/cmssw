///////////////////////////////////////////////////////////////////////////////
// File: HFShower.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShower.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"

#include "G4NavigationHistory.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VSolid.hh"
#include "Randomize.hh"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

#include <iostream>

HFShower::HFShower(const std::string &name,
                   const HcalDDDSimConstants *hcons,
                   const HcalSimulationParameters *hps,
                   edm::ParameterSet const &p,
                   int chk)
    : hcalConstant_(hcons), chkFibre_(chk) {
  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  applyFidCut_ = m_HF.getParameter<bool>("ApplyFiducialCut");
  probMax_ = m_HF.getParameter<double>("ProbMax");

  edm::LogVerbatim("HFShower") << "HFShower:: Maximum probability cut off " << probMax_ << " Check flag " << chkFibre_;

  cherenkov_ = std::make_unique<HFCherenkov>(m_HF);
  fibre_ = std::make_unique<HFFibre>(name, hcalConstant_, hps, p);

  //Special Geometry parameters
  gpar_ = hcalConstant_->getGparHF();
}

HFShower::~HFShower() {}

std::vector<HFShower::Hit> HFShower::getHits(const G4Step *aStep, double weight) {
  std::vector<HFShower::Hit> hits;
  int nHit = 0;

  double edep = weight * (aStep->GetTotalEnergyDeposit());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits: energy " << aStep->GetTotalEnergyDeposit() << " weight " << weight
                               << " edep " << edep;
#endif
  double stepl = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }
  const G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();

  HFShower::Hit hit;
  double energy = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta = momentum / energy;
  double dose = 0.;
  int npeDose = 0;

  const G4ThreeVector &momentumDir = aParticle->GetMomentumDirection();
  const G4ParticleDefinition *particleDef = aTrack->GetDefinition();

  const G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector &globalPos = preStepPoint->GetPosition();
  G4String name = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  //double zv = std::abs(globalPos.z()) - gpar_[4] - 0.5*gpar_[1];
  double zv = std::abs(globalPos.z()) - gpar_[4];
  G4ThreeVector localPos = G4ThreeVector(globalPos.x(), globalPos.y(), zv);
  G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed  (Fibers all long in Geometry!)
  int depth = 1;
  int npmt = 0;
  bool ok = true;
  if (zv < 0. || zv > gpar_[1]) {
    ok = false;  // beyond the fiber in Z
  }
  if (ok && applyFidCut_) {
    npmt = HFFibreFiducial::PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) {  // a short fibre
      if (zv > gpar_[0]) {
        depth = 2;
      } else {
        ok = false;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower: npmt " << npmt << " zv " << std::abs(globalPos.z()) << ":" << gpar_[4]
                                 << ":" << zv << ":" << gpar_[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth = (preStepPoint->GetTouchable()->GetReplicaNumber(0)) % 10;  // All LONG!
  }
  G4ThreeVector translation = preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u = localMom.x();
  double v = localMom.y();
  double w = localMom.z();
  double zCoor = localPos.z();
  double zFibre = (0.5 * gpar_[1] - zCoor - translation.z());
  double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time = fibre_->tShift(localPos, depth, chkFibre_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits: in " << name << " Z " << zCoor << "(" << globalPos.z() << ") "
                               << zFibre << " Trans " << translation << " Time " << tSlice << " " << time
                               << "\n                  Direction " << momentumDir << " Local " << localMom;
#endif
  // Here npe should be 0 if there is no fiber (@@ M.K.)
  int npe = 1;
  std::vector<double> wavelength;
  std::vector<double> momz;
  if (!applyFidCut_) {  // _____ Tmp close of the cherenkov function
    if (ok)
      npe = cherenkov_->computeNPE(aStep, particleDef, pBeta, u, v, w, stepl, zFibre, dose, npeDose);
    wavelength = cherenkov_->getWL();
    momz = cherenkov_->getMom();
  }  // ^^^^^ End of Tmp close of the cherenkov function
  if (ok && npe > 0) {
    for (int i = 0; i < npe; ++i) {
      double p = 1.;
      if (!applyFidCut_)
        p = fibre_->attLength(wavelength[i]);
      double r1 = G4UniformRand();
      double r2 = G4UniformRand();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HFShower") << "HFShower::getHits: " << i << " attenuation " << r1 << ":" << exp(-p * zFibre)
                                   << " r2 " << r2 << ":" << probMax_
                                   << " Survive: " << (r1 <= exp(-p * zFibre) && r2 <= probMax_);
#endif
      if (applyFidCut_ || chkFibre_ < 0 || (r1 <= exp(-p * zFibre) && r2 <= probMax_)) {
        hit.depth = depth;
        hit.time = tSlice + time;
        if (!applyFidCut_) {
          hit.wavelength = wavelength[i];  // Tmp
          hit.momentum = momz[i];          // Tmp
        } else {
          hit.wavelength = 300.;  // Tmp
          hit.momentum = 1.;      // Tmp
        }
        hit.position = globalPos;
        hits.push_back(hit);
        nHit++;
      }
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
  for (int i = 0; i < nHit; ++i)
    edm::LogVerbatim("HFShower") << "HFShower::Hit " << i << " WaveLength " << hits[i].wavelength << " Time "
                                 << hits[i].time << " Momentum " << hits[i].momentum << " Position "
                                 << hits[i].position;
#endif
  return hits;
}

std::vector<HFShower::Hit> HFShower::getHits(const G4Step *aStep, bool forLibraryProducer, double zoffset) {
  std::vector<HFShower::Hit> hits;
  int nHit = 0;

  double edep = aStep->GetTotalEnergyDeposit();
  double stepl = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }
  const G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();

  HFShower::Hit hit;
  double energy = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta = momentum / energy;
  double dose = 0.;
  int npeDose = 0;

  const G4ThreeVector &momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();

  G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector &globalPos = preStepPoint->GetPosition();
  G4String name = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  //double zv = std::abs(globalPos.z()) - gpar_[4] - 0.5*gpar_[1];
  //double zv = std::abs(globalPos.z()) - gpar_[4];
  double zv = gpar_[1] - (std::abs(globalPos.z()) - zoffset);
  G4ThreeVector localPos = G4ThreeVector(globalPos.x(), globalPos.y(), zv);
  G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed  (Fibers all long in Geometry!)
  int depth = 1;
  int npmt = 0;
  bool ok = true;
  if (zv < 0. || zv > gpar_[1]) {
    ok = false;  // beyond the fiber in Z
  }
  if (ok && applyFidCut_) {
    npmt = HFFibreFiducial::PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) {  // a short fibre
      if (zv > gpar_[0]) {
        depth = 2;
      } else {
        ok = false;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower: npmt " << npmt << " zv " << std::abs(globalPos.z()) << ":" << gpar_[4]
                                 << ":" << zv << ":" << gpar_[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth = (preStepPoint->GetTouchable()->GetReplicaNumber(0)) % 10;  // All LONG!
  }
  G4ThreeVector translation = preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u = localMom.x();
  double v = localMom.y();
  double w = localMom.z();
  double zCoor = localPos.z();
  double zFibre = (0.5 * gpar_[1] - zCoor - translation.z());
  double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time = fibre_->tShift(localPos, depth, chkFibre_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits: in " << name << " Z " << zCoor << "(" << globalPos.z() << ") "
                               << zFibre << " Trans " << translation << " Time " << tSlice << " " << time
                               << "\n                  Direction " << momentumDir << " Local " << localMom;
#endif
  // Here npe should be 0 if there is no fiber (@@ M.K.)
  int npe = 1;
  std::vector<double> wavelength;
  std::vector<double> momz;
  if (!applyFidCut_) {  // _____ Tmp close of the cherenkov function
    if (ok)
      npe = cherenkov_->computeNPE(aStep, particleDef, pBeta, u, v, w, stepl, zFibre, dose, npeDose);
    wavelength = cherenkov_->getWL();
    momz = cherenkov_->getMom();
  }  // ^^^^^ End of Tmp close of the cherenkov function
  if (ok && npe > 0) {
    for (int i = 0; i < npe; ++i) {
      double p = 1.;
      if (!applyFidCut_)
        p = fibre_->attLength(wavelength[i]);
      double r1 = G4UniformRand();
      double r2 = G4UniformRand();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HFShower") << "HFShower::getHits: " << i << " attenuation " << r1 << ":" << exp(-p * zFibre)
                                   << " r2 " << r2 << ":" << probMax_
                                   << " Survive: " << (r1 <= exp(-p * zFibre) && r2 <= probMax_);
#endif
      if (applyFidCut_ || chkFibre_ < 0 || (r1 <= exp(-p * zFibre) && r2 <= probMax_)) {
        hit.depth = depth;
        hit.time = tSlice + time;
        if (!applyFidCut_) {
          hit.wavelength = wavelength[i];  // Tmp
          hit.momentum = momz[i];          // Tmp
        } else {
          hit.wavelength = 300.;  // Tmp
          hit.momentum = 1.;      // Tmp
        }
        hit.position = globalPos;
        hits.push_back(hit);
        nHit++;
      }
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
  for (int i = 0; i < nHit; ++i)
    edm::LogVerbatim("HFShower") << "HFShower::Hit " << i << " WaveLength " << hits[i].wavelength << " Time "
                                 << hits[i].time << " Momentum " << hits[i].momentum << " Position "
                                 << hits[i].position;
#endif
  return hits;
}

std::vector<HFShower::Hit> HFShower::getHits(const G4Step *aStep, bool forLibrary) {
  std::vector<HFShower::Hit> hits;
  int nHit = 0;

  double edep = aStep->GetTotalEnergyDeposit();
  double stepl = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }

  const G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();

  HFShower::Hit hit;
  double energy = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta = momentum / energy;
  double dose = 0.;
  int npeDose = 0;

  const G4ThreeVector &momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();

  const G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector &globalPos = preStepPoint->GetPosition();
  G4String name = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  double zv = std::abs(globalPos.z()) - gpar_[4] - 0.5 * gpar_[1];
  G4ThreeVector localPos = G4ThreeVector(globalPos.x(), globalPos.y(), zv);
  G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed (Fibers are all long in Geometry!)
  int depth = 1;
  int npmt = 0;
  bool ok = true;
  if (zv < 0 || zv > gpar_[1]) {
    ok = false;  // beyond the fiber in Z
  }
  if (ok && applyFidCut_) {
    npmt = HFFibreFiducial::PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) {  // a short fibre
      if (zv > gpar_[0]) {
        depth = 2;
      } else {
        ok = false;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShower:getHits(SL): npmt " << npmt << " zv " << std::abs(globalPos.z()) << ":"
                                 << gpar_[4] << ":" << zv << ":" << gpar_[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth = (preStepPoint->GetTouchable()->GetReplicaNumber(0)) % 10;  // All LONG!
  }
  G4ThreeVector translation = preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u = localMom.x();
  double v = localMom.y();
  double w = localMom.z();
  double zCoor = localPos.z();
  double zFibre = (0.5 * gpar_[1] - zCoor - translation.z());
  double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time = fibre_->tShift(localPos, depth, chkFibre_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShower::getHits(SL): in " << name << " Z " << zCoor << "(" << globalPos.z() << ") "
                               << zFibre << " Trans " << translation << " Time " << tSlice << " " << time
                               << "\n                  Direction " << momentumDir << " Local " << localMom;
#endif
  // npe should be 0
  int npe = 0;
  if (ok)
    npe = cherenkov_->computeNPE(aStep, particleDef, pBeta, u, v, w, stepl, zFibre, dose, npeDose);
  std::vector<double> wavelength = cherenkov_->getWL();
  std::vector<double> momz = cherenkov_->getMom();

  for (int i = 0; i < npe; ++i) {
    hit.time = tSlice + time;
    hit.wavelength = wavelength[i];
    hit.momentum = momz[i];
    hit.position = globalPos;
    hits.push_back(hit);
    nHit++;
  }

  return hits;
}
