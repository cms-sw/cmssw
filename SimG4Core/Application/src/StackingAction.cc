#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"
#include "G4EmProcessSubType.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4VSolid.hh"
#include "G4TransportationManager.hh"
#include "G4GammaGeneralProcess.hh"
#include "G4LossTableManager.hh"

StackingAction::StackingAction(const TrackingAction* trka, const edm::ParameterSet& p, const CMSSteppingVerbose* sv)
    : trackAction(trka), steppingVerbose(sv) {
  trackNeutrino = p.getParameter<bool>("TrackNeutrino");
  killHeavy = p.getParameter<bool>("KillHeavy");
  killGamma = p.getParameter<bool>("KillGamma");
  kmaxGamma = p.getParameter<double>("GammaThreshold") * CLHEP::MeV;
  kmaxIon = p.getParameter<double>("IonThreshold") * CLHEP::MeV;
  kmaxProton = p.getParameter<double>("ProtonThreshold") * CLHEP::MeV;
  kmaxNeutron = p.getParameter<double>("NeutronThreshold") * CLHEP::MeV;
  killDeltaRay = p.getParameter<bool>("KillDeltaRay");
  limitEnergyForVacuum = p.getParameter<double>("CriticalEnergyForVacuum") * CLHEP::MeV;
  maxTrackTime = p.getParameter<double>("MaxTrackTime") * ns;
  maxTrackTimeForward = p.getParameter<double>("MaxTrackTimeForward") * ns;
  maxZCentralCMS = p.getParameter<double>("MaxZCentralCMS") * CLHEP::m;
  maxTrackTimes = p.getParameter<std::vector<double> >("MaxTrackTimes");
  maxTimeNames = p.getParameter<std::vector<std::string> >("MaxTimeNames");
  deadRegionNames = p.getParameter<std::vector<std::string> >("DeadRegions");
  savePDandCinAll = p.getUntrackedParameter<bool>("SaveAllPrimaryDecayProductsAndConversions", true);
  savePDandCinTracker = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInTracker", false);
  savePDandCinCalo = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInCalo", false);
  savePDandCinMuon = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInMuon", false);
  saveFirstSecondary = p.getUntrackedParameter<bool>("SaveFirstLevelSecondary", false);

  gRusRoEnerLim = p.getParameter<double>("RusRoGammaEnergyLimit") * CLHEP::MeV;
  nRusRoEnerLim = p.getParameter<double>("RusRoNeutronEnergyLimit") * CLHEP::MeV;

  gRusRoEcal = p.getParameter<double>("RusRoEcalGamma");
  gRusRoHcal = p.getParameter<double>("RusRoHcalGamma");
  gRusRoMuonIron = p.getParameter<double>("RusRoMuonIronGamma");
  gRusRoPreShower = p.getParameter<double>("RusRoPreShowerGamma");
  gRusRoCastor = p.getParameter<double>("RusRoCastorGamma");
  gRusRoWorld = p.getParameter<double>("RusRoWorldGamma");

  nRusRoEcal = p.getParameter<double>("RusRoEcalNeutron");
  nRusRoHcal = p.getParameter<double>("RusRoHcalNeutron");
  nRusRoMuonIron = p.getParameter<double>("RusRoMuonIronNeutron");
  nRusRoPreShower = p.getParameter<double>("RusRoPreShowerNeutron");
  nRusRoCastor = p.getParameter<double>("RusRoCastorNeutron");
  nRusRoWorld = p.getParameter<double>("RusRoWorldNeutron");

  if (gRusRoEnerLim > 0.0 && (gRusRoEcal < 1.0 || gRusRoHcal < 1.0 || gRusRoMuonIron < 1.0 || gRusRoPreShower < 1.0 ||
                              gRusRoCastor < 1.0 || gRusRoWorld < 1.0)) {
    gRRactive = true;
  }
  if (nRusRoEnerLim > 0.0 && (nRusRoEcal < 1.0 || nRusRoHcal < 1.0 || nRusRoMuonIron < 1.0 || nRusRoPreShower < 1.0 ||
                              nRusRoCastor < 1.0 || nRusRoWorld < 1.0)) {
    nRRactive = true;
  }

  if (p.exists("TestKillingOptions")) {
    killInCalo = (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCalo");
    killInCaloEfH = (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCaloEfH");
    edm::LogWarning("SimG4CoreApplication")
        << " *** Activating special test killing options in StackingAction \n"
        << " *** Kill secondaries in Calorimetetrs volume = " << killInCalo << "\n"
        << " *** Kill electromagnetic secondaries from hadrons in Calorimeters volume= " << killInCaloEfH;
  }

  initPointer();
  newTA = new NewTrackAction();

  edm::LogVerbatim("SimG4CoreApplication")
      << "StackingAction initiated with"
      << " flag for saving decay products in "
      << " Tracker: " << savePDandCinTracker << " in Calo: " << savePDandCinCalo << " in Muon: " << savePDandCinMuon
      << " everywhere: " << savePDandCinAll << "\n  saveFirstSecondary"
      << ": " << saveFirstSecondary << " Tracking neutrino flag: " << trackNeutrino
      << " Kill Delta Ray flag: " << killDeltaRay << " Kill hadrons/ions flag: " << killHeavy
      << " MaxZCentralCMS = " << maxZCentralCMS / CLHEP::m << " m"
      << " MaxTrackTimeForward = " << maxTrackTimeForward / CLHEP::ns << " ns";

  if (killHeavy) {
    edm::LogVerbatim("SimG4CoreApplication") << "StackingAction kill protons below " << kmaxProton / CLHEP::MeV
                                             << " MeV, neutrons below " << kmaxNeutron / CLHEP::MeV << " MeV and ions"
                                             << " below " << kmaxIon / CLHEP::MeV << " MeV";
  }
  killExtra = killDeltaRay || killHeavy || killInCalo || killInCaloEfH;

  edm::LogVerbatim("SimG4CoreApplication") << "StackingAction kill tracks with "
                                           << "time larger than " << maxTrackTime / CLHEP::ns << " ns ";
  numberTimes = maxTimeNames.size();
  if (0 < numberTimes) {
    for (unsigned int i = 0; i < numberTimes; ++i) {
      edm::LogVerbatim("SimG4CoreApplication")
          << "          MaxTrackTime for " << maxTimeNames[i] << " is " << maxTrackTimes[i] << " ns ";
      maxTrackTimes[i] *= CLHEP::ns;
    }
  }
  if (limitEnergyForVacuum > 0.0) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "StackingAction LowDensity regions - kill if E < " << limitEnergyForVacuum / CLHEP::MeV << " MeV";
    printRegions(lowdensRegions, "LowDensity");
  }
  if (deadRegions.size() > 0.0) {
    edm::LogVerbatim("SimG4CoreApplication") << "StackingAction Dead regions - kill all secondaries ";
    printRegions(deadRegions, "Dead");
  }
  if (gRRactive) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "StackingAction: "
        << "Russian Roulette for gamma Elimit(MeV)= " << gRusRoEnerLim / CLHEP::MeV << "\n"
        << "                 ECAL Prob= " << gRusRoEcal << "\n"
        << "                 HCAL Prob= " << gRusRoHcal << "\n"
        << "             MuonIron Prob= " << gRusRoMuonIron << "\n"
        << "            PreShower Prob= " << gRusRoPreShower << "\n"
        << "               CASTOR Prob= " << gRusRoCastor << "\n"
        << "                World Prob= " << gRusRoWorld;
  }
  if (nRRactive) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "StackingAction: "
        << "Russian Roulette for neutron Elimit(MeV)= " << nRusRoEnerLim / CLHEP::MeV << "\n"
        << "                 ECAL Prob= " << nRusRoEcal << "\n"
        << "                 HCAL Prob= " << nRusRoHcal << "\n"
        << "             MuonIron Prob= " << nRusRoMuonIron << "\n"
        << "            PreShower Prob= " << nRusRoPreShower << "\n"
        << "               CASTOR Prob= " << nRusRoCastor << "\n"
        << "                World Prob= " << nRusRoWorld;
  }

  if (savePDandCinTracker) {
    edm::LogVerbatim("SimG4CoreApplication") << "StackingAction Tracker regions: ";
    printRegions(trackerRegions, "Tracker");
  }
  if (savePDandCinCalo) {
    edm::LogVerbatim("SimG4CoreApplication") << "StackingAction Calo regions: ";
    printRegions(caloRegions, "Calo");
  }
  if (savePDandCinMuon) {
    edm::LogVerbatim("SimG4CoreApplication") << "StackingAction Muon regions: ";
    printRegions(muonRegions, "Muon");
  }
  worldSolid = G4TransportationManager::GetTransportationManager()
                   ->GetNavigatorForTracking()
                   ->GetWorldVolume()
                   ->GetLogicalVolume()
                   ->GetSolid();
}

StackingAction::~StackingAction() { delete newTA; }

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* aTrack) {
  // G4 interface part
  G4ClassificationOfNewTrack classification = fUrgent;
  const int pdg = aTrack->GetDefinition()->GetPDGEncoding();
  const int abspdg = std::abs(pdg);
  auto track = const_cast<G4Track*>(aTrack);
  const G4VProcess* creatorProc = aTrack->GetCreatorProcess();

  if (creatorProc == nullptr && aTrack->GetParentID() != 0) {
    edm::LogWarning("StackingAction::ClassifyNewTrack")
        << " TrackID=" << aTrack->GetTrackID() << " ParentID=" << aTrack->GetParentID() << " "
        << aTrack->GetDefinition()->GetParticleName() << " Ekin(MeV)=" << aTrack->GetKineticEnergy();
  }
  if (aTrack->GetKineticEnergy() < 0.0) {
    edm::LogWarning("StackingAction::ClassifyNewTrack")
        << " TrackID=" << aTrack->GetTrackID() << " ParentID=" << aTrack->GetParentID() << " "
        << aTrack->GetDefinition()->GetParticleName() << " Ekin(MeV)=" << aTrack->GetKineticEnergy() << " creator "
        << creatorProc->GetProcessName();
  }
  // primary
  if (creatorProc == nullptr || aTrack->GetParentID() == 0) {
    if (!trackNeutrino && (abspdg == 12 || abspdg == 14 || abspdg == 16 || abspdg == 18)) {
      classification = fKill;
    } else if (worldSolid->Inside(aTrack->GetPosition()) == kOutside) {
      classification = fKill;
    } else {
      newTA->primary(track);
    }
  } else {
    // secondary
    const G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
    const double time = aTrack->GetGlobalTime();

    // definetly killed tracks
    if (aTrack->GetTrackStatus() == fStopAndKill) {
      classification = fKill;
    } else if (!trackNeutrino && (abspdg == 12 || abspdg == 14 || abspdg == 16 || abspdg == 18)) {
      classification = fKill;

    } else if (std::abs(aTrack->GetPosition().z()) >= maxZCentralCMS) {
      // very forward secondary
      if (time > maxTrackTimeForward) {
        classification = fKill;
      } else {
        const G4Track* mother = trackAction->geant4Track();
        newTA->secondary(track, *mother, 0);
      }

    } else if (isItOutOfTimeWindow(reg, time)) {
      // time window check
      classification = fKill;

    } else {
      // potentially good for tracking
      const double ke = aTrack->GetKineticEnergy();
      G4int subType = (nullptr != creatorProc) ? creatorProc->GetProcessSubType() : 0;
      // VI: this part of code is needed for Geant4 10.7 only
      if (subType == 16) {
        auto ptr = dynamic_cast<const G4GammaGeneralProcess*>(creatorProc);
        if (nullptr != ptr) {
          creatorProc = ptr->GetSelectedProcess();
          if (nullptr == creatorProc) {
            if (nullptr == m_Compton) {
              auto vp = G4LossTableManager::Instance()->GetEmProcessVector();
              for (auto& p : vp) {
                if (fComptonScattering == p->GetProcessSubType()) {
                  m_Compton = p;
                  break;
                }
              }
            }
            creatorProc = m_Compton;
          }
          subType = creatorProc->GetProcessSubType();
          track->SetCreatorProcess(creatorProc);
        }
        if (creatorProc == nullptr) {
          edm::LogWarning("StackingAction::ClassifyNewTrack")
              << " SubType=16 and no creatorProc; TrackID=" << aTrack->GetTrackID()
              << " ParentID=" << aTrack->GetParentID() << " " << aTrack->GetDefinition()->GetParticleName()
              << " Ekin(MeV)=" << ke << " SubType=" << subType;
        }
      }
      // VI - end
      LogDebug("SimG4CoreApplication") << "##StackingAction:Classify Track " << aTrack->GetTrackID() << " Parent "
                                       << aTrack->GetParentID() << " " << aTrack->GetDefinition()->GetParticleName()
                                       << " Ekin(MeV)=" << ke / CLHEP::MeV << " subType=" << subType << " ";

      // kill tracks in specific regions
      if (isThisRegion(reg, deadRegions)) {
        classification = fKill;
      }
      if (classification != fKill && ke <= limitEnergyForVacuum && isThisRegion(reg, lowdensRegions)) {
        classification = fKill;

      } else if (classification != fKill) {
        // very low-energy gamma
        if (pdg == 22 && killGamma && ke < kmaxGamma) {
          classification = fKill;
        }

        // specific track killing - not for production
        if (killExtra && classification != fKill) {
          if (killHeavy && classification != fKill) {
            if (((pdg / 1000000000 == 1) && (((pdg / 10000) % 100) > 0) && (((pdg / 10) % 100) > 0) &&
                 (ke < kmaxIon)) ||
                ((pdg == 2212) && (ke < kmaxProton)) || ((pdg == 2112) && (ke < kmaxNeutron))) {
              classification = fKill;
            }
          }

          if (killDeltaRay && classification != fKill && subType == fIonisation) {
            classification = fKill;
          }
          if (killInCalo && classification != fKill && isThisRegion(reg, caloRegions)) {
            classification = fKill;
          }
          if (killInCaloEfH && classification != fKill) {
            int pdgMother = std::abs(trackAction->geant4Track()->GetDefinition()->GetPDGEncoding());
            if ((pdg == 22 || abspdg == 11) && pdgMother != 11 && pdgMother != 22 && isThisRegion(reg, caloRegions)) {
              classification = fKill;
            }
          }
        }

        // Russian roulette && MC truth
        if (classification != fKill) {
          const G4Track* mother = trackAction->geant4Track();
          int flag = 0;
          if (savePDandCinAll) {
            flag = isItPrimaryDecayProductOrConversion(subType, *mother);
          } else {
            if ((savePDandCinTracker && isThisRegion(reg, trackerRegions)) ||
                (savePDandCinCalo && isThisRegion(reg, caloRegions)) ||
                (savePDandCinMuon && isThisRegion(reg, muonRegions))) {
              flag = isItPrimaryDecayProductOrConversion(subType, *mother);
            }
          }
          if (saveFirstSecondary && 0 == flag) {
            flag = isItFromPrimary(*mother, flag);
          }

          // Russian roulette
          if (2112 == pdg || 22 == pdg) {
            double currentWeight = aTrack->GetWeight();

            if (1.0 >= currentWeight) {
              double prob = 1.0;
              double elim = 0.0;

              // neutron
              if (nRRactive && pdg == 2112) {
                elim = nRusRoEnerLim;
                if (reg == regionEcal) {
                  prob = nRusRoEcal;
                } else if (reg == regionHcal) {
                  prob = nRusRoHcal;
                } else if (reg == regionMuonIron) {
                  prob = nRusRoMuonIron;
                } else if (reg == regionPreShower) {
                  prob = nRusRoPreShower;
                } else if (reg == regionCastor) {
                  prob = nRusRoCastor;
                } else if (reg == regionWorld) {
                  prob = nRusRoWorld;
                }

                // gamma
              } else if (gRRactive && pdg == 22) {
                elim = gRusRoEnerLim;
                if (reg == regionEcal || reg == regionPreShower) {
                  if (rrApplicable(aTrack, *mother)) {
                    if (reg == regionEcal) {
                      prob = gRusRoEcal;
                    } else {
                      prob = gRusRoPreShower;
                    }
                  }
                } else {
                  if (reg == regionHcal) {
                    prob = gRusRoHcal;
                  } else if (reg == regionMuonIron) {
                    prob = gRusRoMuonIron;
                  } else if (reg == regionCastor) {
                    prob = gRusRoCastor;
                  } else if (reg == regionWorld) {
                    prob = gRusRoWorld;
                  }
                }
              }
              if (prob < 1.0 && aTrack->GetKineticEnergy() < elim) {
                if (G4UniformRand() < prob) {
                  track->SetWeight(currentWeight / prob);
                } else {
                  classification = fKill;
                }
              }
            }
          }
          if (classification != fKill) {
            newTA->secondary(track, *mother, flag);
          }
          LogDebug("SimG4CoreApplication")
              << "StackingAction:Classify Track " << aTrack->GetTrackID() << " Parent " << aTrack->GetParentID()
              << " Type " << aTrack->GetDefinition()->GetParticleName() << " Ekin=" << ke / CLHEP::MeV
              << " MeV from process subType=" << subType << " as " << classification << " Flag: " << flag;
        }
      }
    }
  }
  if (nullptr != steppingVerbose) {
    steppingVerbose->stackFilled(aTrack, (classification == fKill));
  }
  return classification;
}

void StackingAction::NewStage() {}

void StackingAction::PrepareNewEvent() {}

void StackingAction::initPointer() {
  // prepare region vector
  const unsigned int num = maxTimeNames.size();
  maxTimeRegions.resize(num, nullptr);

  // Russian roulette
  const std::vector<G4Region*>* rs = G4RegionStore::GetInstance();

  for (auto& reg : *rs) {
    const G4String& rname = reg->GetName();
    if ((gRusRoEcal < 1.0 || nRusRoEcal < 1.0) && rname == "EcalRegion") {
      regionEcal = reg;
    }
    if ((gRusRoHcal < 1.0 || nRusRoHcal < 1.0) && rname == "HcalRegion") {
      regionHcal = reg;
    }
    if ((gRusRoMuonIron < 1.0 || nRusRoMuonIron < 1.0) && rname == "MuonIron") {
      regionMuonIron = reg;
    }
    if ((gRusRoPreShower < 1.0 || nRusRoPreShower < 1.0) && rname == "PreshowerRegion") {
      regionPreShower = reg;
    }
    if ((gRusRoCastor < 1.0 || nRusRoCastor < 1.0) && rname == "CastorRegion") {
      regionCastor = reg;
    }
    if ((gRusRoWorld < 1.0 || nRusRoWorld < 1.0) && rname == "DefaultRegionForTheWorld") {
      regionWorld = reg;
    }

    // time limits
    for (unsigned int i = 0; i < num; ++i) {
      if (rname == (G4String)(maxTimeNames[i])) {
        maxTimeRegions[i] = reg;
        break;
      }
    }
    //
    if (savePDandCinTracker &&
        (rname == "BeamPipe" || rname == "BeamPipeVacuum" || rname == "TrackerPixelSensRegion" ||
         rname == "TrackerPixelDeadRegion" || rname == "TrackerDeadRegion" || rname == "TrackerSensRegion" ||
         rname == "FastTimerRegionBTL" || rname == "FastTimerRegionETL" || rname == "FastTimerRegionSensBTL" ||
         rname == "FastTimerRegionSensETL")) {
      trackerRegions.push_back(reg);
    }
    if (savePDandCinCalo && (rname == "HcalRegion" || rname == "EcalRegion" || rname == "PreshowerSensRegion" ||
                             rname == "PreshowerRegion" || rname == "APDRegion" || rname == "HGCalRegion")) {
      caloRegions.push_back(reg);
    }
    if (savePDandCinMuon && (rname == "MuonChamber" || rname == "MuonSensitive_RPC" || rname == "MuonIron" ||
                             rname == "Muon" || rname == "MuonSensitive_DT-CSC")) {
      muonRegions.push_back(reg);
    }
    if (rname == "BeamPipeOutside" || rname == "BeamPipeVacuum") {
      lowdensRegions.push_back(reg);
    }
    for (auto& dead : deadRegionNames) {
      if (rname == (G4String)(dead)) {
        deadRegions.push_back(reg);
      }
    }
  }
}

bool StackingAction::isThisRegion(const G4Region* reg, std::vector<const G4Region*>& regions) const {
  bool flag = false;
  for (auto& region : regions) {
    if (reg == region) {
      flag = true;
      break;
    }
  }
  return flag;
}

int StackingAction::isItPrimaryDecayProductOrConversion(const int stype, const G4Track& mother) const {
  int flag = 0;
  const TrackInformation& motherInfo(extractor(mother));
  // Check whether mother is a primary
  if (motherInfo.isPrimary()) {
    if (stype == fDecay) {
      flag = 1;
    } else if (stype == fGammaConversion) {
      flag = 2;
    }
  }
  return flag;
}

bool StackingAction::rrApplicable(const G4Track* aTrack, const G4Track& mother) const {
  const TrackInformation& motherInfo(extractor(mother));

  // Check whether mother is gamma, e+, e-
  const int genID = motherInfo.genParticlePID();
  return (22 != genID && 11 != std::abs(genID));
}

int StackingAction::isItFromPrimary(const G4Track& mother, int flagIn) const {
  int flag = flagIn;
  if (flag != 1) {
    const TrackInformation* ptr = static_cast<TrackInformation*>(mother.GetUserInformation());
    if (ptr->isPrimary()) {
      flag = 3;
    }
  }
  return flag;
}

bool StackingAction::isItOutOfTimeWindow(const G4Region* reg, const double& t) const {
  double tofM = maxTrackTime;
  for (unsigned int i = 0; i < numberTimes; ++i) {
    if (reg == maxTimeRegions[i]) {
      tofM = maxTrackTimes[i];
      break;
    }
  }
  return (t > tofM);
}

void StackingAction::printRegions(const std::vector<const G4Region*>& reg, const std::string& word) const {
  for (unsigned int i = 0; i < reg.size(); ++i) {
    edm::LogVerbatim("SimG4CoreApplication")
        << " StackingAction: " << word << "Region " << i << ". " << reg[i]->GetName();
  }
}
