//
// V.Ivanchenko 2013/10/19
// step limiter and killer for e+,e- and other charged particles
//
#include "SimG4Core/Application/interface/ElectronLimiter.h"
#include "SimG4Core/PhysicsLists/interface/CMSTrackingCutModel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ParticleDefinition.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4LossTableManager.hh"
#include "G4DummyModel.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Region.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationProcessType.hh"

ElectronLimiter::ElectronLimiter(const edm::ParameterSet &p, const G4ParticleDefinition *part)
    : G4VEmProcess("eLimiter", fGeneral),
      ionisation_(nullptr),
      particle_(part),
      nRegions_(0),
      rangeCheckFlag_(false),
      fieldCheckFlag_(false),
      killTrack_(false) {
  // set Process Sub Type
  SetProcessSubType(static_cast<int>(STEP_LIMITER));
  minStepLimit_ = p.getParameter<double>("MinStepLimit") * CLHEP::mm;
  trcut_ = new CMSTrackingCutModel(part);
}

ElectronLimiter::~ElectronLimiter() { delete trcut_; }

void ElectronLimiter::InitialiseProcess(const G4ParticleDefinition *) {
  G4LossTableManager *lManager = G4LossTableManager::Instance();
  if (rangeCheckFlag_) {
    ionisation_ = lManager->GetEnergyLossProcess(particle_);
    if (!ionisation_) {
      rangeCheckFlag_ = false;
    }
  }
  AddEmModel(0, new G4DummyModel());

  if (lManager->IsMaster()) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "ElectronLimiter::BuildPhysicsTable for " << particle_->GetParticleName() << " ioni: " << ionisation_
        << " rangeCheckFlag: " << rangeCheckFlag_ << " fieldCheckFlag: " << fieldCheckFlag_ << " " << nRegions_
        << " regions for tracking cuts\n";
  }
}

G4double ElectronLimiter::PostStepGetPhysicalInteractionLength(const G4Track &aTrack,
                                                               G4double previousLimit,
                                                               G4ForceCondition *condition) {
  *condition = NotForced;
  G4double limit = DBL_MAX;
  killTrack_ = false;

  G4double kinEnergy = aTrack.GetKineticEnergy();
  if (0 < nRegions_) {
    if (regions_[0] == nullptr) {
      if (kinEnergy < energyLimits_[0]) {
        killTrack_ = true;
        trcut_->InitialiseForStep(factors_[0], rms_[0]);
        limit = 0.0;
      }
    } else {
      const G4Region *reg = aTrack.GetVolume()->GetLogicalVolume()->GetRegion();
      for (G4int i = 0; i < nRegions_; ++i) {
        if (reg == regions_[i] && kinEnergy < energyLimits_[i]) {
          killTrack_ = true;
          trcut_->InitialiseForStep(factors_[i], rms_[i]);
          limit = 0.0;
          break;
        }
      }
    }
  }
  if (!killTrack_ && rangeCheckFlag_) {
    G4double safety = aTrack.GetStep()->GetPreStepPoint()->GetSafety();
    if (safety > std::min(minStepLimit_, previousLimit)) {
      G4double range = ionisation_->GetRangeForLoss(kinEnergy, aTrack.GetMaterialCutsCouple());
      if (safety >= range) {
        killTrack_ = true;
        limit = 0.0;
      }
    }
  }
  if (!killTrack_ && fieldCheckFlag_) {
    limit = minStepLimit_;
  }
  return limit;
}

G4VParticleChange *ElectronLimiter::PostStepDoIt(const G4Track &aTrack, const G4Step &) {
  fParticleChange.Initialize(aTrack);
  if (killTrack_) {
    fParticleChange.ProposeTrackStatus(fStopAndKill);
    G4double edep = trcut_->SampleEnergyDepositEcal(aTrack.GetKineticEnergy());
    fParticleChange.ProposeLocalEnergyDeposit(edep);
    fParticleChange.SetProposedKineticEnergy(0.0);
  }
  return &fParticleChange;
}

G4bool ElectronLimiter::IsApplicable(const G4ParticleDefinition &) { return true; }

void ElectronLimiter::StartTracking(G4Track *) {}

void ElectronLimiter::SetTrackingCutPerRegion(std::vector<const G4Region *> &reg,
                                              std::vector<G4double> &cut,
                                              std::vector<G4double> &fac,
                                              std::vector<G4double> &rms) {
  nRegions_ = reg.size();
  if (nRegions_ > 0) {
    regions_ = reg;
    energyLimits_ = cut;
    factors_ = fac;
    rms_ = rms;
  }
}
