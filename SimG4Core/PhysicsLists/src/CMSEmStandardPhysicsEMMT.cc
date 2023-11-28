#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsEMMT.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsTrackingManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4EmParameters.hh"
#include "G4EmBuilder.hh"

#include "G4hMultipleScattering.hh"
#include "G4ionIonisation.hh"

#include "G4ParticleTable.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

#include "G4MuonNuclearProcess.hh"
#include "G4MuonVDNuclearModel.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"

CMSEmStandardPhysicsEMMT::CMSEmStandardPhysicsEMMT(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_emmt"), fParameterSet(p) {
  SetVerboseLevel(ver);
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetApplyCuts(true);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  param->SetFluo(false);
  SetPhysicsType(bElectromagnetic);
  double tcut = p.getParameter<double>("G4TrackingCut") * CLHEP::MeV;
  param->SetLowestElectronEnergy(tcut);
  param->SetLowestMuHadEnergy(tcut);
}

CMSEmStandardPhysicsEMMT::~CMSEmStandardPhysicsEMMT() {}

void CMSEmStandardPhysicsEMMT::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysicsEMMT::ConstructProcess() {
  if (verboseLevel > 0) {
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct EM Processes";
  }

  G4EmBuilder::PrepareEMPhysics();

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  // processes used by several particles
  G4hMultipleScattering* hmsc = new G4hMultipleScattering("ionmsc");
  G4NuclearStopping* pnuc(nullptr);

  // register specialized tracking for e-/e+ and gammas
  auto* trackingManager = new CMSEmStandardPhysicsTrackingManager(fParameterSet);
  G4Electron::Electron()->SetTrackingManager(trackingManager);
  G4Positron::Positron()->SetTrackingManager(trackingManager);
  G4Gamma::Gamma()->SetTrackingManager(trackingManager);

  // generic ion
  G4ParticleDefinition* particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);

  // muons, hadrons ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);

  // add muon-nuclear processes (normally done by G4EmExtraPhysics)
  G4MuonNuclearProcess* muNucProcess = new G4MuonNuclearProcess();
  G4MuonVDNuclearModel* muNucModel = new G4MuonVDNuclearModel();
  muNucProcess->RegisterMe(muNucModel);
  ph->RegisterProcess(muNucProcess, G4MuonPlus::MuonPlus());
  ph->RegisterProcess(muNucProcess, G4MuonMinus::MuonMinus());
}
