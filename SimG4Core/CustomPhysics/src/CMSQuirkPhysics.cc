#include "SimG4Core/CustomPhysics/interface/CMSQuirkPhysics.h"
#include "SimG4Core/CustomPhysics/interface/Quirk.h"
#include "SimG4Core/CustomPhysics/interface/QuirkStringStore.h"
#include "SimG4Core/CustomPhysics/interface/QuirkTransportation.h"
#include "SimG4Core/CustomPhysics/interface/QuirkWatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4EventManager.hh"
#include "G4StackManager.hh"
#include "G4Threading.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4hBremsstrahlung.hh"
#include "G4hPairProduction.hh"
#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/PhysicalConstants.h>

namespace {
  // string tension F = Lambda^2 / hbar c
  double stringForce(double lambda) { return lambda * lambda / CLHEP::hbarc; }
}  // namespace

CMSQuirkPhysics::CMSQuirkPhysics(const edm::ParameterSet& p)
    : G4VPhysicsConstructor("Quirk Physics"),
      m_verbose(p.getUntrackedParameter<int>("QuirkVerbose", 0)),
      m_pdg(std::abs(p.getParameter<int>("QuirkPDGID"))),
      m_mass(p.getParameter<double>("QuirkMass") * CLHEP::GeV),
      m_charge(p.getParameter<double>("QuirkCharge") * CLHEP::eplus),
      m_stringForce(stringForce(p.getParameter<double>("QuirkLambda") * CLHEP::eV)),
      m_firstStringLength(p.getParameter<double>("QuirkFirstStringLength") * CLHEP::mm),
      m_maxBoost(p.getParameter<double>("QuirkMaxBoost")),
      m_maxMergeT(p.getParameter<double>("QuirkMaxMerge") * CLHEP::mm),
      m_maxMergeMag(m_maxMergeT),
      m_looperTrials(p.getParameter<int>("QuirkLooperTrials")),
      m_rMax(p.getParameter<double>("QuirkMaxRadius") * CLHEP::cm),
      m_zMax(p.getParameter<double>("QuirkMaxZ") * CLHEP::cm),
      m_keepStopped(p.getParameter<bool>("QuirkKeepStopped")),
      m_moveTinySteps(p.getParameter<bool>("QuirkMoveTinySteps")),
      // off by default: G4MuPairProductionModel has no element data for PDG 17 (em0033 above 8 m_Q)
      m_pairProduction(p.getParameter<bool>("QuirkPairProduction")),
      // must stay well below the first string, else the restart repeats at production
      m_crossingLength(std::min(p.getParameter<double>("QuirkCrossingLength") * CLHEP::mm, 0.1 * m_firstStringLength)),
      m_stopThreshold(p.getParameter<double>("QuirkStopThreshold") * CLHEP::MeV) {
  if (m_mass <= 0. || m_stringForce <= 0. || m_maxBoost <= 0. || m_maxBoost >= 1.) {
    throw cms::Exception("Configuration")
        << "CMSQuirkPhysics: invalid parameters QuirkMass= " << m_mass / CLHEP::GeV
        << " GeV, QuirkLambda= " << p.getParameter<double>("QuirkLambda") << " eV, QuirkMaxBoost= " << m_maxBoost;
  }
  edm::LogVerbatim("SimG4CoreCustomPhysics")
      << "CMSQuirkPhysics: PDG " << m_pdg << " mass " << m_mass / CLHEP::GeV << " GeV, charge "
      << m_charge / CLHEP::eplus << " e, Lambda " << std::sqrt(m_stringForce * CLHEP::hbarc) / CLHEP::eV
      << " eV (string force " << m_stringForce / (CLHEP::MeV / CLHEP::mm) << " MeV/mm), first string length "
      << m_firstStringLength / CLHEP::mm << " mm, max boost " << m_maxBoost << ", max merge " << m_maxMergeT / CLHEP::mm
      << " mm, killed at r > " << m_rMax / CLHEP::cm << " cm or |z| > " << m_zMax / CLHEP::cm
      << " cm (<= 0: never), pair production " << (m_pairProduction ? "on" : "off") << ", stopped quirks "
      << (m_keepStopped ? "kept while the partner moves" : "killed");
}

void CMSQuirkPhysics::ConstructParticle() {
  // definitions are created once, on the master
  if (nullptr != m_quirk) {
    return;
  }
  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  for (int pdg : {m_pdg, -m_pdg}) {
    const G4ParticleDefinition* old = table->FindParticle(pdg);
    if (nullptr != old) {
      throw cms::Exception("Configuration") << "CMSQuirkPhysics: PDG code " << pdg << " is already defined as "
                                            << old->GetParticleName() << "; do not combine with other exotica physics";
    }
  }
  m_quirk = new Quirk("quirk", m_mass, m_charge, m_pdg);
  m_antiQuirk = new Quirk("antiquirk", m_mass, -m_charge, -m_pdg);
}

void CMSQuirkPhysics::ConstructProcess() {
  // string state of this thread
  QuirkStringStore::instance().configure(m_stringForce, m_firstStringLength, m_maxBoost, m_maxMergeT, m_maxMergeMag);

  for (Quirk* q : {m_quirk, m_antiQuirk}) {
    G4ProcessManager* pmanager = q->GetProcessManager();
    if (nullptr == pmanager) {
      throw cms::Exception("Configuration") << "CMSQuirkPhysics: quirk without a process manager";
    }
    // replace whatever was attached by the generic constructors
    while (pmanager->GetProcessListLength() != 0) {
      pmanager->RemoveProcess(0);
    }
    auto transport = new QuirkTransportation(m_verbose, m_moveTinySteps, m_crossingLength);
    transport->SetThresholdTrials(m_looperTrials);
    pmanager->AddProcess(transport, -1, 0, 0);
    pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
    pmanager->AddProcess(new G4hIonisation, -1, 2, 2);
    pmanager->AddProcess(new G4hBremsstrahlung, -1, 3, 3);
    if (m_pairProduction) {
      pmanager->AddProcess(new G4hPairProduction, -1, 4, 4);
    }
    pmanager->AddProcess(new QuirkWatcher(m_rMax, m_zMax, m_keepStopped, m_stopThreshold), -1, -1, 5);
    if (m_verbose > 1) {
      pmanager->DumpInfo();
    }
  }

  // the two quirks alternate through one extra waiting stack; user stacking actions keep
  // these defaults for suspended tracks
  if (!G4Threading::IsMasterThread()) {
    G4StackManager* stack = G4EventManager::GetEventManager()->GetStackManager();
    stack->SetNumberOfAdditionalWaitingStacks(1);
    stack->SetDefaultClassification(fSuspendAndWait, fWaiting);
    stack->SetDefaultClassification(fSuspend, fWaiting_1);
  }
}
