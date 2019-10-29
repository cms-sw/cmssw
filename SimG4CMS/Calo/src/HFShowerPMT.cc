///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPMT.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerPMT.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4NavigationHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <sstream>

//#define EDM_ML_DEBUG

HFShowerPMT::HFShowerPMT(const std::string& name,
                         const HcalDDDSimConstants* hcons,
                         const HcalSimulationParameters* hps,
                         edm::ParameterSet const& p)
    : hcalConstant_(hcons), hcalsimpar_(hps) {
  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShowerPMT");
  pePerGeV = m_HF.getParameter<double>("PEPerGeVPMT");

  //Special Geometry parameters
  pmtR1 = hcalsimpar_->pmtRight_;
  pmtFib1 = hcalsimpar_->pmtFiberRight_;
  pmtR2 = hcalsimpar_->pmtLeft_;
  pmtFib2 = hcalsimpar_->pmtFiberLeft_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerPMT: gets the Index matches for " << pmtR1.size() << " PMTs";
  for (unsigned int ii = 0; ii < pmtR1.size(); ii++) {
    edm::LogVerbatim("HFShower") << "HFShowerPMT: rIndexR[" << ii << "] = " << pmtR1[ii] << " fibreR[" << ii
                                 << "] = " << pmtFib1[ii] << " rIndexL[" << ii << "] = " << pmtR2[ii] << " fibreL["
                                 << ii << "] = " << pmtFib2[ii];
  }
#endif
  cherenkov_ = std::make_unique<HFCherenkov>(m_HF);

  // Special Geometry parameters
  rTable = hcalConstant_->getRTableHF();
#ifdef EDM_ML_DEBUG
  std::stringstream sss;
  for (unsigned int ig = 0; ig < rTable.size(); ++ig) {
    if (ig / 10 * 10 == ig) {
      sss << "\n";
    }
    sss << "  " << rTable[ig] / cm;
  }
  edm::LogVerbatim("HFShowerPMT") << "HFShowerPMT: " << rTable.size() << " rTable(cm):" << sss.str();
#endif
}

HFShowerPMT::~HFShowerPMT() {}

double HFShowerPMT::getHits(const G4Step* aStep) {
  indexR = indexF = -1;

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  int boxNo = touch->GetReplicaNumber(2);
  int pmtNo = touch->GetReplicaNumber(1);
  if (boxNo <= 1) {
    indexR = pmtR1[pmtNo - 1];
    indexF = pmtFib1[pmtNo - 1];
  } else {
    indexR = pmtR2[pmtNo - 1];
    indexF = pmtFib2[pmtNo - 1];
  }

#ifdef EDM_ML_DEBUG
  double edep = aStep->GetTotalEnergyDeposit();
  edm::LogVerbatim("HFShower") << "HFShowerPMT: Box " << boxNo << " PMT " << pmtNo << " Mapped Indices " << indexR
                               << ", " << indexF << " Edeposit " << edep / MeV << " MeV; PE " << edep * pePerGeV / GeV;
#endif

  double photons = 0;
  if (indexR >= 0 && indexF > 0) {
    G4Track* aTrack = aStep->GetTrack();
    G4ParticleDefinition* particleDef = aTrack->GetDefinition();
    double stepl = aStep->GetStepLength();
    double beta = preStepPoint->GetBeta();
    G4ThreeVector pDir = aTrack->GetDynamicParticle()->GetMomentumDirection();
    G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformAxis(pDir);
    photons = cherenkov_->computeNPEinPMT(particleDef, beta, localMom.x(), localMom.y(), localMom.z(), stepl);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShowerPMT::getHits: for particle " << particleDef->GetParticleName() << " Step "
                                 << stepl << " Beta " << beta << " Direction " << pDir << " Local " << localMom
                                 << " p.e. " << photons;
#endif
  }
  return photons;
}

double HFShowerPMT::getRadius() {
  double r = 0.;
  if (indexR >= 0 && indexR + 1 < (int)(rTable.size()))
    r = 0.5 * (rTable[indexR] + rTable[indexR + 1]);
#ifdef EDM_ML_DEBUG
  else
    edm::LogVerbatim("HFShower") << "HFShowerPMT::getRadius: R " << indexR << " F " << indexF;
#endif
  if (indexF == 2)
    r = -r;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerPMT: Radius (" << indexR << "/" << indexF << ") " << r;
#endif
  return r;
}
