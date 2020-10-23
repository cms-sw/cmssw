//
// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang
// V.Ivanchenko rename the class, cleanup, and move
//              to SimG4Core/Application - 2012/08/14

#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/GFlashEMShowerModel.h"
#include "SimG4Core/Application/interface/GFlashHadronShowerModel.h"
#include "SimG4Core/Application/interface/LowEnergyFastSimModel.h"
#include "SimG4Core/Application/interface/ElectronLimiter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4FastSimulationManagerProcess.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4RegionStore.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"

#include "G4EmParameters.hh"
#include "G4LossTableManager.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProcessManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Transportation.hh"
#include "G4UAtomicDeexcitation.hh"
#include <memory>

#include <string>
#include <vector>

const G4int NREG = 7;
const G4String rname[NREG] = {"EcalRegion",
                              "HcalRegion",
                              "HGcalRegion",
                              "MuonIron",
                              "PreshowerRegion",
                              "CastorRegion",
                              "DefaultRegionForTheWorld"};

struct ParametrisedEMPhysics::TLSmod {
  std::unique_ptr<GFlashEMShowerModel> theEcalEMShowerModel;
  std::unique_ptr<LowEnergyFastSimModel> theLowEnergyFastSimModel;
  std::unique_ptr<GFlashEMShowerModel> theHcalEMShowerModel;
  std::unique_ptr<GFlashHadronShowerModel> theEcalHadShowerModel;
  std::unique_ptr<GFlashHadronShowerModel> theHcalHadShowerModel;
  std::unique_ptr<G4FastSimulationManagerProcess> theFastSimulationManagerProcess;
};

G4ThreadLocal ParametrisedEMPhysics::TLSmod* ParametrisedEMPhysics::m_tpmod = nullptr;

ParametrisedEMPhysics::ParametrisedEMPhysics(const std::string& name, const edm::ParameterSet& p)
    : G4VPhysicsConstructor(name), theParSet(p) {
  // bremsstrahlung threshold and EM verbosity
  G4EmParameters* param = G4EmParameters::Instance();
  G4int verb = theParSet.getUntrackedParameter<int>("Verbosity", 0);
  param->SetVerbose(verb);

  G4double bremth = theParSet.getParameter<double>("G4BremsstrahlungThreshold") * GeV;
  param->SetBremsstrahlungTh(bremth);

  bool fluo = theParSet.getParameter<bool>("FlagFluo");
  param->SetFluo(fluo);

  bool modifyT = theParSet.getParameter<bool>("ModifyTransportation");
  double th1 = theParSet.getUntrackedParameter<double>("ThresholdWarningEnergy") * MeV;
  double th2 = theParSet.getUntrackedParameter<double>("ThresholdImportantEnergy") * MeV;
  int nt = theParSet.getUntrackedParameter<int>("ThresholdTrials");

  edm::LogVerbatim("SimG4CoreApplication")
      << "ParametrisedEMPhysics::ConstructProcess: bremsstrahlung threshold Eth= " << bremth / GeV << " GeV"
      << "\n                                         verbosity= " << verb << "  fluoFlag: " << fluo
      << "  modifyTransport: " << modifyT << "  Ntrials= " << nt
      << "\n                                         ThWarning(MeV)= " << th1 << "  ThException(MeV)= " << th2;

  // Russian roulette and tracking cut for e+-
  double energyLim = theParSet.getParameter<double>("RusRoElectronEnergyLimit") * MeV;
  if (energyLim > 0.0) {
    G4double rrfact[NREG] = {1.0};

    rrfact[0] = theParSet.getParameter<double>("RusRoEcalElectron");
    rrfact[1] = theParSet.getParameter<double>("RusRoHcalElectron");
    rrfact[2] = theParSet.getParameter<double>("RusRoMuonIronElectron");
    rrfact[3] = theParSet.getParameter<double>("RusRoPreShowerElectron");
    rrfact[4] = theParSet.getParameter<double>("RusRoCastorElectron");
    rrfact[5] = theParSet.getParameter<double>("RusRoWorldElectron");
    for (int i = 0; i < NREG; ++i) {
      if (rrfact[i] < 1.0) {
        param->ActivateSecondaryBiasing("eIoni", rname[i], rrfact[i], energyLim);
        param->ActivateSecondaryBiasing("hIoni", rname[i], rrfact[i], energyLim);
        edm::LogVerbatim("SimG4CoreApplication")
            << "ParametrisedEMPhysics: Russian Roulette"
            << " for e- Prob= " << rrfact[i] << " Elimit(MeV)= " << energyLim / CLHEP::MeV << " inside " << rname[i];
      }
    }
  }
}

ParametrisedEMPhysics::~ParametrisedEMPhysics() {
  if (m_tpmod) {
    delete m_tpmod;
    m_tpmod = nullptr;
  }
}

void ParametrisedEMPhysics::ConstructParticle() {
  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();
}

void ParametrisedEMPhysics::ConstructProcess() {
  edm::LogVerbatim("SimG4CoreApplication") << "ParametrisedEMPhysics::ConstructProcess() started";

  // GFlash part
  bool gem = theParSet.getParameter<bool>("GflashEcal");
  bool lowEnergyGem = theParSet.getParameter<bool>("LowEnergyGflashEcal");
  bool ghad = theParSet.getParameter<bool>("GflashHcal");
  bool gemHad = theParSet.getParameter<bool>("GflashEcalHad");
  bool ghadHad = theParSet.getParameter<bool>("GflashHcalHad");

  if (gem || ghad || lowEnergyGem || gemHad || ghadHad) {
    if (!m_tpmod) {
      m_tpmod = new TLSmod;
    }
    edm::LogVerbatim("SimG4CoreApplication")
        << "ParametrisedEMPhysics: GFlash Construct for e+-: " << gem << "  " << ghad << " " << lowEnergyGem
        << " for hadrons: " << gemHad << "  " << ghadHad;

    m_tpmod->theFastSimulationManagerProcess = std::make_unique<G4FastSimulationManagerProcess>();

    if (gem || ghad) {
      G4Electron::Electron()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4Positron::Positron()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
    } else if (lowEnergyGem) {
      G4Electron::Electron()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4Positron::Positron()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
    }

    if (gemHad || ghadHad) {
      G4Proton::Proton()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4AntiProton::AntiProton()->GetProcessManager()->AddDiscreteProcess(
          m_tpmod->theFastSimulationManagerProcess.get());
      G4PionPlus::PionPlus()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4PionMinus::PionMinus()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4KaonPlus::KaonPlus()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
      G4KaonMinus::KaonMinus()->GetProcessManager()->AddDiscreteProcess(m_tpmod->theFastSimulationManagerProcess.get());
    }

    if (gem || gemHad || lowEnergyGem) {
      G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("EcalRegion", false);

      if (!aRegion) {
        edm::LogWarning("SimG4CoreApplication") << "ParametrisedEMPhysics::ConstructProcess: "
                                                << "EcalRegion is not defined, GFlash will not be enabled for ECAL!";

      } else {
        if (gem) {
          //Electromagnetic Shower Model for ECAL
          m_tpmod->theEcalEMShowerModel =
              std::make_unique<GFlashEMShowerModel>("GflashEcalEMShowerModel", aRegion, theParSet);
        } else if (lowEnergyGem) {
          //Low energy electromagnetic Shower Model for ECAL
          m_tpmod->theLowEnergyFastSimModel =
              std::make_unique<LowEnergyFastSimModel>("LowEnergyFastSimModel", aRegion, theParSet);
        }

        if (gemHad) {
          //Electromagnetic Shower Model for ECAL
          m_tpmod->theEcalHadShowerModel =
              std::make_unique<GFlashHadronShowerModel>("GflashEcalHadShowerModel", aRegion, theParSet);
        }
      }
    }
    if (ghad || ghadHad) {
      G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
      if (!aRegion) {
        edm::LogWarning("SimG4CoreApplication") << "ParametrisedEMPhysics::ConstructProcess: "
                                                << "HcalRegion is not defined, GFlash will not be enabled for HCAL!";

      } else {
        if (ghad) {
          //Electromagnetic Shower Model for HCAL
          m_tpmod->theHcalEMShowerModel =
              std::make_unique<GFlashEMShowerModel>("GflashHcalEMShowerModel", aRegion, theParSet);
        }
        if (ghadHad) {
          //Electromagnetic Shower Model for ECAL
          m_tpmod->theHcalHadShowerModel =
              std::make_unique<GFlashHadronShowerModel>("GflashHcalHadShowerModel", aRegion, theParSet);
        }
      }
    }
  }

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Step limiters for e+-
  bool eLimiter = theParSet.getParameter<bool>("ElectronStepLimit");
  bool rLimiter = theParSet.getParameter<bool>("ElectronRangeTest");
  bool pLimiter = theParSet.getParameter<bool>("PositronStepLimit");
  // Step limiters for hadrons
  bool pTCut = theParSet.getParameter<bool>("ProtonRegionLimit");
  bool piTCut = theParSet.getParameter<bool>("PionRegionLimit");

  std::vector<std::string> regnames = theParSet.getParameter<std::vector<std::string> >("LimitsPerRegion");
  std::vector<double> limitsE = theParSet.getParameter<std::vector<double> >("EnergyLimitsE");
  std::vector<double> limitsH = theParSet.getParameter<std::vector<double> >("EnergyLimitsH");
  std::vector<double> facE = theParSet.getParameter<std::vector<double> >("EnergyFactorsE");
  std::vector<double> rmsE = theParSet.getParameter<std::vector<double> >("EnergyRMSE");
  int nlimits = regnames.size();
  int nlimitsH = 0;
  std::vector<const G4Region*> reg;
  std::vector<G4double> rlimE;
  std::vector<G4double> rlimH;
  std::vector<G4double> factE;
  std::vector<G4double> rmsvE;
  if (0 < nlimits) {
    G4RegionStore* store = G4RegionStore::GetInstance();
    for (int i = 0; i < nlimits; ++i) {
      // apply limiter for whole CMS
      if (regnames[i] == "all") {
        reg.clear();
        rlimE.clear();
        rlimH.clear();
        factE.clear();
        rmsvE.clear();
        reg.emplace_back(nullptr);
        rlimE.emplace_back(limitsE[i] * CLHEP::MeV);
        rlimH.emplace_back(limitsH[i] * CLHEP::MeV);
        factE.emplace_back(facE[i]);
        rmsvE.emplace_back(rmsE[i]);
        nlimitsH = (limitsH[i] > 0) ? 1 : 0;
        break;
      }
      const G4Region* r = store->GetRegion(regnames[i], false);
      // apply for concrete G4Region
      if (r && (limitsE[i] > 0.0 || limitsH[i] > 0.0)) {
        reg.emplace_back(r);
        rlimE.emplace_back(limitsE[i] * CLHEP::MeV);
        rlimH.emplace_back(limitsH[i] * CLHEP::MeV);
        factE.emplace_back(facE[i]);
        rmsvE.emplace_back(rmsE[i]);
        if (limitsH[i] > 0) {
          ++nlimitsH;
        }
      }
    }
    nlimits = reg.size();
  }

  if (eLimiter || rLimiter || 0 < nlimits) {
    ElectronLimiter* elim = new ElectronLimiter(theParSet, G4Electron::Electron());
    elim->SetRangeCheckFlag(rLimiter);
    elim->SetFieldCheckFlag(eLimiter);
    elim->SetTrackingCutPerRegion(reg, rlimE, factE, rmsvE);
    ph->RegisterProcess(elim, G4Electron::Electron());
  }

  if (pLimiter || 0 < nlimits) {
    ElectronLimiter* plim = new ElectronLimiter(theParSet, G4Positron::Positron());
    plim->SetFieldCheckFlag(pLimiter);
    plim->SetTrackingCutPerRegion(reg, rlimE, factE, rmsvE);
    ph->RegisterProcess(plim, G4Positron::Positron());
  }
  if (0 < nlimits && 0 < nlimitsH) {
    if (pTCut) {
      ElectronLimiter* plim = new ElectronLimiter(theParSet, G4Proton::Proton());
      plim->SetFieldCheckFlag(pLimiter);
      plim->SetTrackingCutPerRegion(reg, rlimH, factE, rmsvE);
      ph->RegisterProcess(plim, G4Proton::Proton());
    }
    if (piTCut) {
      ElectronLimiter* plim = new ElectronLimiter(theParSet, G4PionPlus::PionPlus());
      plim->SetFieldCheckFlag(pLimiter);
      plim->SetTrackingCutPerRegion(reg, rlimH, factE, rmsvE);
      ph->RegisterProcess(plim, G4PionPlus::PionPlus());
      plim = new ElectronLimiter(theParSet, G4PionMinus::PionMinus());
      plim->SetFieldCheckFlag(pLimiter);
      plim->SetTrackingCutPerRegion(reg, rlimH, factE, rmsvE);
      ph->RegisterProcess(plim, G4PionMinus::PionMinus());
    }
  }
  // enable fluorescence
  bool fluo = theParSet.getParameter<bool>("FlagFluo");
  if (fluo && !G4LossTableManager::Instance()->AtomDeexcitation()) {
    G4VAtomDeexcitation* de = new G4UAtomicDeexcitation();
    G4LossTableManager::Instance()->SetAtomDeexcitation(de);
  }
  // change parameters of transportation
  bool modifyT = theParSet.getParameter<bool>("ModifyTransportation");
  if (modifyT) {
    double th1 = theParSet.getUntrackedParameter<double>("ThresholdWarningEnergy") * MeV;
    double th2 = theParSet.getUntrackedParameter<double>("ThresholdImportantEnergy") * MeV;
    int nt = theParSet.getUntrackedParameter<int>("ThresholdTrials");
    ModifyTransportation(G4Electron::Electron(), nt, th1, th2);
  }
  edm::LogVerbatim("SimG4CoreApplication") << "ParametrisedEMPhysics::ConstructProcess() is done";
}

void ParametrisedEMPhysics::ModifyTransportation(const G4ParticleDefinition* part, int ntry, double th1, double th2) {
  G4ProcessManager* man = part->GetProcessManager();
  G4Transportation* trans = (G4Transportation*)((*(man->GetProcessList()))[0]);
  if (trans) {
    trans->SetThresholdWarningEnergy(th1);
    trans->SetThresholdImportantEnergy(th2);
    trans->SetThresholdTrials(ntry);
    edm::LogVerbatim("SimG4CoreApplication")
        << "ParametrisedEMPhysics: printout level changed for " << part->GetParticleName();
  }
}
