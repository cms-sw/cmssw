#include "SimG4Core/CheckSecondary/interface/TreatSecondary.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

TreatSecondary::TreatSecondary(const edm::ParameterSet &p) : typeEnumerator(nullptr) {
  verbosity = p.getUntrackedParameter<int>("Verbosity", 0);
  killAfter = p.getUntrackedParameter<int>("KillAfter", -1);
  minDeltaE = p.getUntrackedParameter<double>("MinimumDeltaE", 10.0) * MeV;

  edm::LogInfo("CheckSecondary") << "Instantiate CheckSecondary with Flag"
                                 << " for Killing track after " << killAfter
                                 << " hadronic interactions\nDefine inelastic"
                                 << " if > 1 seondary or change in KE > " << minDeltaE << " MeV\n";

  typeEnumerator = new G4ProcessTypeEnumerator();
}

TreatSecondary::~TreatSecondary() {
  if (typeEnumerator)
    delete typeEnumerator;
}

void TreatSecondary::initTrack(const G4Track *thTk) {
  step = 0;
  nsecL = 0;
  nHad = 0;
  eTrack = thTk->GetKineticEnergy() / MeV;
  LogDebug("CheckSecondary") << "TreatSecondary::initTrack:Track: " << thTk->GetTrackID()
                             << " Type: " << thTk->GetDefinition()->GetParticleName() << " KE "
                             << thTk->GetKineticEnergy() / GeV << " GeV p " << thTk->GetMomentum().mag() / GeV
                             << " GeV daughter of particle " << thTk->GetParentID();
}

std::vector<math::XYZTLorentzVector> TreatSecondary::tracks(
    const G4Step *aStep, std::string &name, int &procid, bool &hadrInt, double &deltaE, std::vector<int> &charges) {
  step++;
  procid = -1;
  name = "Unknown";
  hadrInt = false;
  deltaE = 0;
  std::vector<math::XYZTLorentzVector> secondaries;
  charges.clear();

  if (aStep != nullptr) {
    const G4TrackVector *tkV = aStep->GetSecondary();
    G4Track *thTk = aStep->GetTrack();
    const G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    const G4StepPoint *postStepPoint = aStep->GetPostStepPoint();
    double eTrackNew = thTk->GetKineticEnergy() / MeV;
    deltaE = eTrack - eTrackNew;
    eTrack = eTrackNew;
    if (tkV != nullptr && postStepPoint != nullptr) {
      int nsc = (*tkV).size();
      const G4VProcess *proc = postStepPoint->GetProcessDefinedStep();
      if (proc != nullptr) {
        G4ProcessType type = proc->GetProcessType();
        procid = typeEnumerator->processIdLong(proc);
        name = proc->GetProcessName();
        int sec = nsc - nsecL;
        LogDebug("CheckSecondary") << sec << " secondaries in step " << thTk->GetCurrentStepNumber() << " of track "
                                   << thTk->GetTrackID() << " from " << name << " of type " << type << " ID " << procid
                                   << " (" << typeEnumerator->processG4Name(procid) << ")";

        // hadronic interaction
        if (procid >= 121 && procid <= 151) {
          LogDebug("CheckSecondary") << "Hadronic Interaction " << nHad << " of Type " << procid << " with " << sec
                                     << " secondaries from process " << proc->GetProcessName() << " Delta E " << deltaE
                                     << " Flag " << hadrInt;
          math::XYZTLorentzVector secondary;
          for (int i = nsecL; i < nsc; ++i) {
            G4Track *tk = (*tkV)[i];
            G4ThreeVector pp = tk->GetMomentum();
            double ee = tk->GetTotalEnergy();
            secondary = math::XYZTLorentzVector(pp.x(), pp.y(), pp.z(), ee);
            secondaries.push_back(secondary);
            int charge = (int)(tk->GetDefinition()->GetPDGCharge());
            charges.push_back(charge);
          }
          if (verbosity > 0) {
            for (int i = nsecL; i < nsc; i++) {
              G4Track *tk = (*tkV)[i];
              LogDebug("CheckSecondary") << "Secondary: " << sec << " ID " << tk->GetTrackID() << " Status "
                                         << tk->GetTrackStatus() << " Particle "
                                         << tk->GetDefinition()->GetParticleName() << " Position " << tk->GetPosition()
                                         << " KE " << tk->GetKineticEnergy() << " Time " << tk->GetGlobalTime();
            }
          }
        }
        nsecL = nsc;
      }
    }
    if (verbosity > 1) {
      LogDebug("CheckSecondary") << "Track: " << thTk->GetTrackID() << " Status " << thTk->GetTrackStatus()
                                 << " Particle " << thTk->GetDefinition()->GetParticleName() << " at "
                                 << preStepPoint->GetPosition() << " Step: " << step << " KE "
                                 << thTk->GetKineticEnergy() / GeV << " GeV; p " << thTk->GetMomentum().mag() / GeV
                                 << " GeV/c; Step Length " << aStep->GetStepLength() << " Energy Deposit "
                                 << aStep->GetTotalEnergyDeposit() / MeV << " MeV; Interaction " << hadrInt;
    }
  }
  return secondaries;
}
