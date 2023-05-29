#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsTrackingManager.h"
#include "TrackingManagerHelper.h"

#include "G4CoulombScattering.hh"
#include "G4UrbanMscModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eCoulombScatteringModel.hh"
#include "G4eIonisation.hh"
#include "G4eMultipleScattering.hh"
#include "G4eplusAnnihilation.hh"

#include "G4ElectroVDNuclearModel.hh"
#include "G4ElectronNuclearProcess.hh"
#include "G4PositronNuclearProcess.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4CascadeInterface.hh"
#include "G4CrossSectionDataSetRegistry.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4GammaNuclearXS.hh"
#include "G4GammaParticipants.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4HadronInelasticProcess.hh"
#include "G4HadronicParameters.hh"
#include "G4LowEGammaNuclearModel.hh"
#include "G4PhotoNuclearCrossSection.hh"
#include "G4QGSMFragmentation.hh"
#include "G4QGSModel.hh"
#include "G4TheoFSGenerator.hh"

#include "G4GammaGeneralProcess.hh"
#include "G4LossTableManager.hh"

#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include <string>

CMSEmStandardPhysicsTrackingManager *CMSEmStandardPhysicsTrackingManager::masterTrackingManager = nullptr;

CMSEmStandardPhysicsTrackingManager::CMSEmStandardPhysicsTrackingManager(const edm::ParameterSet &p) {
  fRangeFactor = p.getParameter<double>("G4MscRangeFactor");
  fGeomFactor = p.getParameter<double>("G4MscGeomFactor");
  fSafetyFactor = p.getParameter<double>("G4MscSafetyFactor");
  fLambdaLimit = p.getParameter<double>("G4MscLambdaLimit") * CLHEP::mm;
  std::string msc = p.getParameter<std::string>("G4MscStepLimit");
  fStepLimitType = fUseSafety;
  if (msc == "UseSafetyPlus") {
    fStepLimitType = fUseSafetyPlus;
  } else if (msc == "Minimal") {
    fStepLimitType = fMinimal;
  }

  G4EmParameters *param = G4EmParameters::Instance();
  G4double highEnergyLimit = param->MscEnergyLimit();

  const G4Region *aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
  const G4Region *bRegion = G4RegionStore::GetInstance()->GetRegion("HGCalRegion", false);

  // e-
  {
    G4eMultipleScattering *msc = new G4eMultipleScattering;
    G4UrbanMscModel *msc1 = new G4UrbanMscModel;
    G4WentzelVIModel *msc2 = new G4WentzelVIModel;
    msc1->SetHighEnergyLimit(highEnergyLimit);
    msc2->SetLowEnergyLimit(highEnergyLimit);
    msc->SetEmModel(msc1);
    msc->SetEmModel(msc2);

    // e-/e+ msc for HCAL and HGCAL using the Urban model
    if (nullptr != aRegion || nullptr != bRegion) {
      G4UrbanMscModel *msc3 = new G4UrbanMscModel();
      msc3->SetHighEnergyLimit(highEnergyLimit);
      msc3->SetRangeFactor(fRangeFactor);
      msc3->SetGeomFactor(fGeomFactor);
      msc3->SetSafetyFactor(fSafetyFactor);
      msc3->SetLambdaLimit(fLambdaLimit);
      msc3->SetStepLimitType(fStepLimitType);
      msc3->SetLocked(true);
      if (nullptr != aRegion) {
        msc->AddEmModel(-1, msc3, aRegion);
      }
      if (nullptr != bRegion) {
        msc->AddEmModel(-1, msc3, bRegion);
      }
    }

    electron.msc = msc;

    electron.ioni = new G4eIonisation;
    electron.brems = new G4eBremsstrahlung;

    G4CoulombScattering *ss = new G4CoulombScattering;
    G4eCoulombScatteringModel *ssm = new G4eCoulombScatteringModel;
    ssm->SetLowEnergyLimit(highEnergyLimit);
    ssm->SetActivationLowEnergyLimit(highEnergyLimit);
    ss->SetEmModel(ssm);
    ss->SetMinKinEnergy(highEnergyLimit);
    electron.ss = ss;
  }

  // e+
  {
    G4eMultipleScattering *msc = new G4eMultipleScattering;
    G4UrbanMscModel *msc1 = new G4UrbanMscModel;
    G4WentzelVIModel *msc2 = new G4WentzelVIModel;
    msc1->SetHighEnergyLimit(highEnergyLimit);
    msc2->SetLowEnergyLimit(highEnergyLimit);
    msc->SetEmModel(msc1);
    msc->SetEmModel(msc2);

    // e-/e+ msc for HCAL and HGCAL using the Urban model
    if (nullptr != aRegion || nullptr != bRegion) {
      G4UrbanMscModel *msc3 = new G4UrbanMscModel();
      msc3->SetHighEnergyLimit(highEnergyLimit);
      msc3->SetRangeFactor(fRangeFactor);
      msc3->SetGeomFactor(fGeomFactor);
      msc3->SetSafetyFactor(fSafetyFactor);
      msc3->SetLambdaLimit(fLambdaLimit);
      msc3->SetStepLimitType(fStepLimitType);
      msc3->SetLocked(true);
      if (nullptr != aRegion) {
        msc->AddEmModel(-1, msc3, aRegion);
      }
      if (nullptr != bRegion) {
        msc->AddEmModel(-1, msc3, bRegion);
      }
    }

    positron.msc = msc;

    positron.ioni = new G4eIonisation;
    positron.brems = new G4eBremsstrahlung;
    positron.annihilation = new G4eplusAnnihilation;

    G4CoulombScattering *ss = new G4CoulombScattering;
    G4eCoulombScatteringModel *ssm = new G4eCoulombScatteringModel;
    ssm->SetLowEnergyLimit(highEnergyLimit);
    ssm->SetActivationLowEnergyLimit(highEnergyLimit);
    ss->SetEmModel(ssm);
    ss->SetMinKinEnergy(highEnergyLimit);
    positron.ss = ss;
  }

  // gamma
  {
    gammaProc = new G4GammaGeneralProcess;

    gammaProc->AddEmProcess(new G4PhotoElectricEffect);
    gammaProc->AddEmProcess(new G4ComptonScattering);
    gammaProc->AddEmProcess(new G4GammaConversion);

    G4HadronInelasticProcess *nuc = new G4HadronInelasticProcess("photonNuclear", G4Gamma::Definition());
    auto xsreg = G4CrossSectionDataSetRegistry::Instance();
    G4VCrossSectionDataSet *xs = nullptr;
    bool useGammaNuclearXS = true;
    if (useGammaNuclearXS) {
      xs = xsreg->GetCrossSectionDataSet("GammaNuclearXS");
      if (nullptr == xs)
        xs = new G4GammaNuclearXS;
    } else {
      xs = xsreg->GetCrossSectionDataSet("PhotoNuclearXS");
      if (nullptr == xs)
        xs = new G4PhotoNuclearCrossSection;
    }
    nuc->AddDataSet(xs);

    G4QGSModel<G4GammaParticipants> *theStringModel = new G4QGSModel<G4GammaParticipants>;
    G4QGSMFragmentation *theFrag = new G4QGSMFragmentation;
    G4ExcitedStringDecay *theStringDecay = new G4ExcitedStringDecay(theFrag);
    theStringModel->SetFragmentationModel(theStringDecay);

    G4GeneratorPrecompoundInterface *theCascade = new G4GeneratorPrecompoundInterface;

    G4TheoFSGenerator *theModel = new G4TheoFSGenerator;
    theModel->SetTransport(theCascade);
    theModel->SetHighEnergyGenerator(theStringModel);

    G4HadronicParameters *param = G4HadronicParameters::Instance();

    G4CascadeInterface *cascade = new G4CascadeInterface;

    // added low-energy model LEND disabled
    G4double gnLowEnergyLimit = 200 * CLHEP::MeV;
    if (gnLowEnergyLimit > 0.0) {
      G4LowEGammaNuclearModel *lemod = new G4LowEGammaNuclearModel;
      lemod->SetMaxEnergy(gnLowEnergyLimit);
      nuc->RegisterMe(lemod);
      cascade->SetMinEnergy(gnLowEnergyLimit - CLHEP::MeV);
    }
    cascade->SetMaxEnergy(param->GetMaxEnergyTransitionFTF_Cascade());
    nuc->RegisterMe(cascade);
    theModel->SetMinEnergy(param->GetMinEnergyTransitionFTF_Cascade());
    theModel->SetMaxEnergy(param->GetMaxEnergy());
    nuc->RegisterMe(theModel);

    gammaProc->AddHadProcess(nuc);
    G4LossTableManager::Instance()->SetGammaGeneralProcess(gammaProc);
  }

  // Create lepto-nuclear processes last, they access cross section data from
  // the gamma nuclear process!
  G4ElectroVDNuclearModel *eModel = new G4ElectroVDNuclearModel;

  {
    G4ElectronNuclearProcess *nuc = new G4ElectronNuclearProcess;
    nuc->RegisterMe(eModel);
    electron.nuc = nuc;
  }
  {
    G4PositronNuclearProcess *nuc = new G4PositronNuclearProcess;
    nuc->RegisterMe(eModel);
    positron.nuc = nuc;
  }

  if (masterTrackingManager == nullptr) {
    masterTrackingManager = this;
  } else {
    electron.msc->SetMasterProcess(masterTrackingManager->electron.msc);
    electron.ss->SetMasterProcess(masterTrackingManager->electron.ss);
    electron.ioni->SetMasterProcess(masterTrackingManager->electron.ioni);
    electron.brems->SetMasterProcess(masterTrackingManager->electron.brems);
    electron.nuc->SetMasterProcess(masterTrackingManager->electron.nuc);

    positron.msc->SetMasterProcess(masterTrackingManager->positron.msc);
    positron.ss->SetMasterProcess(masterTrackingManager->positron.ss);
    positron.ioni->SetMasterProcess(masterTrackingManager->positron.ioni);
    positron.brems->SetMasterProcess(masterTrackingManager->positron.brems);
    positron.annihilation->SetMasterProcess(masterTrackingManager->positron.annihilation);
    positron.nuc->SetMasterProcess(masterTrackingManager->positron.nuc);

    gammaProc->SetMasterProcess(masterTrackingManager->gammaProc);
  }
}

CMSEmStandardPhysicsTrackingManager::~CMSEmStandardPhysicsTrackingManager() {
  if (masterTrackingManager == this) {
    masterTrackingManager = nullptr;
  }
}

void CMSEmStandardPhysicsTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    electron.msc->BuildPhysicsTable(part);
    electron.ioni->BuildPhysicsTable(part);
    electron.brems->BuildPhysicsTable(part);
    electron.ss->BuildPhysicsTable(part);
    electron.nuc->BuildPhysicsTable(part);
  } else if (&part == G4Positron::Definition()) {
    positron.msc->BuildPhysicsTable(part);
    positron.ioni->BuildPhysicsTable(part);
    positron.brems->BuildPhysicsTable(part);
    positron.annihilation->BuildPhysicsTable(part);
    positron.ss->BuildPhysicsTable(part);
    positron.nuc->BuildPhysicsTable(part);
  } else if (&part == G4Gamma::Definition()) {
    gammaProc->BuildPhysicsTable(part);
  }
}

void CMSEmStandardPhysicsTrackingManager::PreparePhysicsTable(const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    electron.msc->PreparePhysicsTable(part);
    electron.ioni->PreparePhysicsTable(part);
    electron.brems->PreparePhysicsTable(part);
    electron.ss->PreparePhysicsTable(part);
    electron.nuc->PreparePhysicsTable(part);
  } else if (&part == G4Positron::Definition()) {
    positron.msc->PreparePhysicsTable(part);
    positron.ioni->PreparePhysicsTable(part);
    positron.brems->PreparePhysicsTable(part);
    positron.annihilation->PreparePhysicsTable(part);
    positron.ss->PreparePhysicsTable(part);
    positron.nuc->PreparePhysicsTable(part);
  } else if (&part == G4Gamma::Definition()) {
    gammaProc->PreparePhysicsTable(part);
  }
}

void CMSEmStandardPhysicsTrackingManager::TrackElectron(G4Track *aTrack) {
  class ElectronPhysics final : public TrackingManagerHelper::Physics {
  public:
    ElectronPhysics(CMSEmStandardPhysicsTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      auto &electron = fMgr.electron;

      electron.msc->StartTracking(aTrack);
      electron.ioni->StartTracking(aTrack);
      electron.brems->StartTracking(aTrack);
      electron.ss->StartTracking(aTrack);
      electron.nuc->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override {
      auto &electron = fMgr.electron;

      electron.msc->EndTracking();
      electron.ioni->EndTracking();
      electron.brems->EndTracking();
      electron.ss->EndTracking();
      electron.nuc->EndTracking();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      auto &electron = fMgr.electron;
      G4double physIntLength, proposedSafety = DBL_MAX;
      G4ForceCondition condition;
      G4GPILSelection selection;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength = electron.nuc->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 3;
      }

      physIntLength = electron.ss->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      physIntLength = electron.brems->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 1;
      }

      physIntLength = electron.ioni->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 2;
      }

      physIntLength =
          electron.ioni->AlongStepGPIL(track, fPreviousStepLength, fProposedStep, proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = -1;
      }

      physIntLength =
          electron.msc->AlongStepGPIL(track, fPreviousStepLength, fProposedStep, proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        // Check if MSC actually wants to win, in most cases it only limits the
        // step size.
        if (selection == CandidateForSelection) {
          fSelected = -1;
        }
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &track, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      auto &electron = fMgr.electron;
      G4VParticleChange *particleChange;

      particleChange = electron.msc->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      particleChange = electron.ioni->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step, G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      auto &electron = fMgr.electron;
      G4VProcess *process = nullptr;
      G4VParticleChange *particleChange = nullptr;

      switch (fSelected) {
        case 3:
          process = electron.nuc;
          particleChange = electron.nuc->PostStepDoIt(track, step);
          break;
        case 0:
          process = electron.ss;
          particleChange = electron.ss->PostStepDoIt(track, step);
          break;
        case 1:
          process = electron.brems;
          particleChange = electron.brems->PostStepDoIt(track, step);
          break;
        case 2:
          process = electron.ioni;
          particleChange = electron.ioni->PostStepDoIt(track, step);
          break;
      }

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    CMSEmStandardPhysicsTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  ElectronPhysics physics(*this);
  TrackingManagerHelper::TrackChargedParticle(aTrack, physics);
}

void CMSEmStandardPhysicsTrackingManager::TrackPositron(G4Track *aTrack) {
  class PositronPhysics final : public TrackingManagerHelper::Physics {
  public:
    PositronPhysics(CMSEmStandardPhysicsTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      auto &positron = fMgr.positron;

      positron.msc->StartTracking(aTrack);
      positron.ioni->StartTracking(aTrack);
      positron.brems->StartTracking(aTrack);
      positron.annihilation->StartTracking(aTrack);
      positron.ss->StartTracking(aTrack);
      positron.nuc->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override {
      auto &positron = fMgr.positron;

      positron.msc->EndTracking();
      positron.ioni->EndTracking();
      positron.brems->EndTracking();
      positron.annihilation->EndTracking();
      positron.ss->EndTracking();
      positron.nuc->EndTracking();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      auto &positron = fMgr.positron;
      G4double physIntLength, proposedSafety = DBL_MAX;
      G4ForceCondition condition;
      G4GPILSelection selection;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength = positron.nuc->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 4;
      }

      physIntLength = positron.ss->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      physIntLength = positron.annihilation->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 1;
      }

      physIntLength = positron.brems->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 2;
      }

      physIntLength = positron.ioni->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 3;
      }

      physIntLength =
          positron.ioni->AlongStepGPIL(track, fPreviousStepLength, fProposedStep, proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = -1;
      }

      physIntLength =
          positron.msc->AlongStepGPIL(track, fPreviousStepLength, fProposedStep, proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        // Check if MSC actually wants to win, in most cases it only limits the
        // step size.
        if (selection == CandidateForSelection) {
          fSelected = -1;
        }
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &track, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      auto &positron = fMgr.positron;
      G4VParticleChange *particleChange;

      particleChange = positron.msc->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      particleChange = positron.ioni->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step, G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      auto &positron = fMgr.positron;
      G4VProcess *process;
      G4VParticleChange *particleChange = nullptr;

      switch (fSelected) {
        case 4:
          process = positron.nuc;
          particleChange = positron.nuc->PostStepDoIt(track, step);
          break;
        case 0:
          process = positron.ss;
          particleChange = positron.ss->PostStepDoIt(track, step);
          break;
        case 1:
          process = positron.annihilation;
          particleChange = positron.annihilation->PostStepDoIt(track, step);
          break;
        case 2:
          process = positron.brems;
          particleChange = positron.brems->PostStepDoIt(track, step);
          break;
        case 3:
          process = positron.ioni;
          particleChange = positron.ioni->PostStepDoIt(track, step);
          break;
      }

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

    G4bool HasAtRestProcesses() override { return true; }

    void AtRestDoIt(G4Track &track, G4Step &step, G4TrackVector &secondaries) override {
      auto &positron = fMgr.positron;
      // Annihilate the positron at rest.
      G4VParticleChange *particleChange = positron.annihilation->AtRestDoIt(track, step);
      particleChange->UpdateStepForAtRest(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(positron.annihilation);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    CMSEmStandardPhysicsTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  PositronPhysics physics(*this);
  TrackingManagerHelper::TrackChargedParticle(aTrack, physics);
}

void CMSEmStandardPhysicsTrackingManager::TrackGamma(G4Track *aTrack) {
  class GammaPhysics final : public TrackingManagerHelper::Physics {
  public:
    GammaPhysics(CMSEmStandardPhysicsTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      fMgr.gammaProc->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override { fMgr.gammaProc->EndTracking(); }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      G4double physIntLength;
      G4ForceCondition condition;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength = fMgr.gammaProc->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step, G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      G4VProcess *process = fMgr.gammaProc;
      G4VParticleChange *particleChange = fMgr.gammaProc->PostStepDoIt(track, step);

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    CMSEmStandardPhysicsTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  GammaPhysics physics(*this);
  TrackingManagerHelper::TrackNeutralParticle(aTrack, physics);
}

void CMSEmStandardPhysicsTrackingManager::HandOverOneTrack(G4Track *aTrack) {
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  if (part == G4Electron::Definition()) {
    TrackElectron(aTrack);
  } else if (part == G4Positron::Definition()) {
    TrackPositron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    TrackGamma(aTrack);
  }

  aTrack->SetTrackStatus(fStopAndKill);
  delete aTrack;
}
