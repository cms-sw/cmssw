#include "DBremWatcher.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4ProductionCuts.hh"
#include "G4ProcessTable.hh"
#include "G4ProcessManager.hh"
#include "G4MuonMinus.hh"
#include "SimG4Core/CustomPhysics/interface/G4muDarkBremsstrahlung.h"
#include "G4Track.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"
#include "G4VParticleChange.hh"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <algorithm>

DBremWatcher::DBremWatcher(edm::ParameterSet const& p) {
  edm::ParameterSet ps = p.getParameter<edm::ParameterSet>("DBremWatcher");
  biasFactor = ps.getUntrackedParameter<double>("DBremBiasFactor", 1);
  m_weight = 0;
  foundbrem = false;
  finaltraj = G4ThreeVector(0, 0, 0);
  aPrimeTraj = G4ThreeVector(0, 0, 0);
  VertexPos = G4ThreeVector(0, 0, 0);
  f_energy = 0;
  produces<float>("DBremEventWeight");
  produces<float>("DBremLocationX");
  produces<float>("DBremLocationY");
  produces<float>("DBremLocationZ");
  //produces<std::string>("DBremMaterial");
  produces<float>("DBremAngle");
  produces<float>("DBremInitialEnergy");
  produces<float>("DBremFinalEnergy");
  produces<float>("BiasFactor");
  pdgs_ = ps.getUntrackedParameter<std::vector<int>>("PDGCodes");
  edm::LogInfo("DBremWatcher") << "DBremWatcher:: Save Sim Track if PDG code "
                               << "is one from the list of " << pdgs_.size() << " items";
  for (unsigned int k = 0; k < pdgs_.size(); ++k)
    edm::LogInfo("DBremWatcher") << "[" << k << "] " << pdgs_[k];
}

DBremWatcher::~DBremWatcher() {}

void DBremWatcher::update(const BeginOfTrack* trk) {
  G4Track* theTrack = (G4Track*)((*trk)());
  TrackInformation* trkInfo = (TrackInformation*)(theTrack->GetUserInformation());
  if (trkInfo) {
    int pdg = theTrack->GetDefinition()->GetPDGEncoding();
    G4ThreeVector Vpos = theTrack->GetVertexPosition();
    const G4VProcess* TrPro = theTrack->GetCreatorProcess();
    if (TrPro != nullptr) {
      if ((theTrack->GetCreatorProcess()->GetProcessName()) == "muDBrem") {
        if (std::find(pdgs_.begin(), pdgs_.end(), pdg) == pdgs_.end()) {
          //Found the deflected muon track
          trkInfo->storeTrack(true);
          if (!theTrack->IsGoodForTracking()) {
            theTrack->SetGoodForTrackingFlag(true);
          }
          f_energy = theTrack->GetTotalEnergy();
          foundbrem = true;
          finaltraj = theTrack->GetMomentum();
        } else {
          m_weight = theTrack->GetWeight();
        }
      }
    }
    if (std::find(pdgs_.begin(), pdgs_.end(), pdg) != pdgs_.end()) {
      //Found an A'
      trkInfo->storeTrack(true);
      VertexPos = Vpos;
      aPrimeTraj = theTrack->GetMomentum();
      LogDebug("DBremWatcher") << "Save SimTrack the Track " << theTrack->GetTrackID() << " Type "
                               << theTrack->GetDefinition()->GetParticleName() << " Momentum "
                               << theTrack->GetMomentum() / MeV << " MeV/c";
    }
  }
}

void DBremWatcher::update(const BeginOfRun* run) {}

void DBremWatcher::update(const BeginOfEvent* event) {
  G4String pname = "muDBrem";
  G4ProcessTable* ptable = G4ProcessTable::GetProcessTable();
  G4bool state = true;
  ptable->SetProcessActivation(pname, state);
  foundbrem = false;
}

void DBremWatcher::update(const EndOfEvent* event) {}

void DBremWatcher::update(const EndOfTrack* trk) {
  G4Track* theTrack = (G4Track*)((*trk)());
  TrackInformation* trkInfo = (TrackInformation*)(theTrack->GetUserInformation());
  const G4VProcess* TrPro = theTrack->GetCreatorProcess();
  if (trkInfo && TrPro != nullptr) {
    int pdg = theTrack->GetDefinition()->GetPDGEncoding();

    if (std::find(pdgs_.begin(), pdgs_.end(), pdg) == pdgs_.end() &&
        (theTrack->GetCreatorProcess()->GetProcessName()) == "muDBrem") {
      trkInfo->storeTrack(true);
    }
  }
}

void DBremWatcher::produce(edm::Event& fEvent, const edm::EventSetup&) {
  if (foundbrem) {
    std::unique_ptr<float> weight = std::make_unique<float>(m_weight);
    fEvent.put(std::move(weight), "DBremEventWeight");
    std::unique_ptr<float> vtxposx = std::make_unique<float>(VertexPos.x());
    std::unique_ptr<float> vtxposy = std::make_unique<float>(VertexPos.y());
    std::unique_ptr<float> vtxposz = std::make_unique<float>(VertexPos.z());
    fEvent.put(std::move(vtxposx), "DBremLocationX");
    fEvent.put(std::move(vtxposy), "DBremLocationY");
    fEvent.put(std::move(vtxposz), "DBremLocationZ");
    std::unique_ptr<float> finalE = std::make_unique<float>(f_energy / GeV);
    fEvent.put(std::move(finalE), "DBremFinalEnergy");
    float deflectionAngle = -1;
    float initialEnergy = sqrt(pow(finaltraj.x() + aPrimeTraj.x(), 2) + pow(finaltraj.y() + aPrimeTraj.y(), 2) +
                               pow(finaltraj.z() + aPrimeTraj.z(), 2) + pow(0.1056 * GeV, 2));
    G4ThreeVector mother(
        finaltraj.x() + aPrimeTraj.x(), finaltraj.y() + aPrimeTraj.y(), finaltraj.z() + aPrimeTraj.z());
    deflectionAngle = mother.angle(finaltraj);
    std::unique_ptr<float> dAngle = std::make_unique<float>(deflectionAngle);
    std::unique_ptr<float> initialE = std::make_unique<float>(initialEnergy / GeV);
    fEvent.put(std::move(dAngle), "DBremAngle");
    fEvent.put(std::move(initialE), "DBremInitialEnergy");
    std::unique_ptr<float> bias = std::make_unique<float>(biasFactor);
    fEvent.put(std::move(bias), "BiasFactor");
  } else {
    std::unique_ptr<float> weight = std::make_unique<float>(0.);
    fEvent.put(std::move(weight), "DBremEventWeight");
  }
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(DBremWatcher);
