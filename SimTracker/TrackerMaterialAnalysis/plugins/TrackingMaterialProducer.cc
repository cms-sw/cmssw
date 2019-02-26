#include <iostream>     // FIXME: switch to MessagLogger & friends
#include <vector>
#include <string>
#include <cassert>
#include <exception>
#include <tuple>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// GEANT4
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VSolid.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4AffineTransform.hh"

#include "TrackingMaterialProducer.h"

// Uncomment the following #define directive to have the full list of
// volumes known to G4 printed to LogInfo("TrackingMaterialProducer")

#define DEBUG_G4_VOLUMES

using namespace CLHEP;
using edm::LogInfo;

// missing from GEANT4 < 9.0 : G4LogicalVolumeStore::GetVolume( name )
static
const G4LogicalVolume* GetVolume(const std::string& name) {
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();

#ifdef DEBUG_G4_VOLUMES
  for (G4LogicalVolumeStore::const_iterator volume = lvs->begin(); volume != lvs->end(); ++volume)
    LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: G4 registered volumes "
                                         << (*volume)->GetName() << std::endl;
#endif

  for (G4LogicalVolumeStore::const_iterator volume = lvs->begin(); volume != lvs->end(); ++volume) {
    if ((const std::string&) (*volume)->GetName() == name)
      return (*volume);
  }
  return nullptr;
}

// missing from GEANT4 : G4TouchableHistory::GetTransform( depth )
static inline
const G4AffineTransform& GetTransform(const G4TouchableHistory* touchable, int depth)
{
  return touchable->GetHistory()->GetTransform( touchable->GetHistory()->GetDepth() - depth );
}

// navigate up the hierarchy of volumes until one with an attached sensitive detector is found
// return a tuple holding
//   - pointer to the first (deepest) sensitive G4VPhysicalVolume
//   - how may steps up in the hierarchy it is (0 is the starting volume)
// if no sensitive detector is found, return a NULL pointer and 0

std::tuple<const G4VPhysicalVolume*, int> GetSensitiveVolume( const G4VTouchable* touchable )
{
  int depth = touchable->GetHistoryDepth();
  for (int level = 0; level < depth; ++level) {      // 0 is self
    const G4VPhysicalVolume* volume = touchable->GetVolume(level);
    if (volume->GetLogicalVolume()->GetSensitiveDetector() != nullptr) {
      return std::make_tuple(volume, level);
    }
  }
  return std::tuple<const G4VPhysicalVolume*, int>(nullptr, 0);
}

//-------------------------------------------------------------------------
TrackingMaterialProducer::TrackingMaterialProducer(const edm::ParameterSet& iPSet)
{
  edm::ParameterSet config = iPSet.getParameter<edm::ParameterSet>("TrackingMaterialProducer");
  m_selectedNames       = config.getParameter< std::vector<std::string> >("SelectedVolumes");
  m_primaryTracks       = config.getParameter<bool>("PrimaryTracksOnly");
  m_tracks              = nullptr;

  produces< std::vector<MaterialAccountingTrack> >();
  output_file_ = new TFile("radLen_vs_eta_fromProducer.root", "RECREATE");
  output_file_->cd();
  radLen_vs_eta_ = new TProfile("radLen", "radLen", 250., -5., 5., 0, 10.);
}

//-------------------------------------------------------------------------
TrackingMaterialProducer::~TrackingMaterialProducer(void)
{
}

//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const EndOfJob* event)
{
  radLen_vs_eta_->Write();
  output_file_->Close();
}
//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const BeginOfJob* event)
{
  // INFO
  LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: List of the selected volumes: " << std::endl;
  for (std::vector<std::string>::const_iterator volume_name = m_selectedNames.begin();
       volume_name != m_selectedNames.end(); ++volume_name) {
    const G4LogicalVolume* volume = GetVolume(*volume_name);
    if (volume) {
      LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: " << *volume_name << std::endl;
      m_selectedVolumes.push_back( volume );
    } else {
      // FIXME: throw an exception ?
      std::cerr << "TrackingMaterialProducer::update(const BeginOfJob*): WARNING: selected volume \"" << *volume_name << "\" not found in geometry " << std::endl;
    }
  }
}


//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const BeginOfEvent* event)
{
  m_tracks = new std::vector<MaterialAccountingTrack>();
}


//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const BeginOfTrack* event)
{
  m_track.reset();

  // prevent secondary tracks from propagating
  G4Track* track = const_cast<G4Track*>((*event)());
  if (m_primaryTracks and track->GetParentID() != 0) {
    track->SetTrackStatus(fStopAndKill);
  }
}

bool TrackingMaterialProducer::isSelectedFast(const G4TouchableHistory* touchable) {
  for (int d = touchable->GetHistoryDepth() -1; d >=0;  --d) {
      if (
           std::find(
                     m_selectedNames.begin(),
                     m_selectedNames.end(),
                     touchable->GetVolume(d)->GetName())
        != m_selectedNames.end())
        return true;
    }
  return false;
}

//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const G4Step* step)
{
  const G4TouchableHistory* touchable = static_cast<const G4TouchableHistory*>(step->GetTrack()->GetTouchable());
  if (not isSelectedFast( touchable )) {
    LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer:\t[...] skipping "
                                         << touchable->GetVolume()->GetName() << std::endl;
    return;
  }

  // material and step proterties
  const G4Material* material = touchable->GetVolume()->GetLogicalVolume()->GetMaterial();
  double length = step->GetStepLength() / cm;          // mm -> cm
  double X0 = material->GetRadlen() / cm;              // mm -> cm
  double Ne = material->GetElectronDensity() * cm3;    // 1/mm3 -> 1/cm3
  double Xi = Ne / 6.0221415e23 * 0.307075 / 2;        // MeV / cm
  double radiationLengths = length / X0;               //
  double energyLoss       = length * Xi / 1000.;       // GeV
  //double energyLoss = step->GetDeltaEnergy()/MeV;  should we use this??

  G4ThreeVector globalPosPre  = step->GetPreStepPoint()->GetPosition();
  G4ThreeVector globalPosPost = step->GetPostStepPoint()->GetPosition();
  GlobalPoint globalPositionIn(  globalPosPre.x()  / cm, globalPosPre.y()  / cm, globalPosPre.z() / cm );    // mm -> cm
  GlobalPoint globalPositionOut( globalPosPost.x() / cm, globalPosPost.y() / cm, globalPosPost.z() / cm );   // mm -> cm

  // check for a sensitive detector
  bool enter_sensitive = false;
  bool leave_sensitive = false;
  double cosThetaPre  = 0.0;
  double cosThetaPost = 0.0;
  int level = 0;
  const G4VPhysicalVolume* sensitive = nullptr;
  GlobalPoint position;
  std::tie(sensitive, level) = GetSensitiveVolume(touchable);
  if (sensitive) {
    const G4VSolid &          solid     = *touchable->GetSolid( level );
    const G4AffineTransform & transform = GetTransform( touchable, level );
    G4ThreeVector pos = transform.Inverse().TransformPoint( G4ThreeVector( 0., 0., 0. ) );
    position = GlobalPoint( pos.x() / cm, pos.y() / cm, pos.z() / cm );  // mm -> cm

    G4ThreeVector localPosPre   = transform.TransformPoint( globalPosPre );
    EInside       statusPre     = solid.Inside( localPosPre );
    if (statusPre == kSurface) {
      enter_sensitive = true;
      G4ThreeVector globalDirPre  = step->GetPreStepPoint()->GetMomentumDirection();
      G4ThreeVector localDirPre   = transform.TransformAxis( globalDirPre );
      G4ThreeVector normalPre     = solid.SurfaceNormal( localPosPre );
      cosThetaPre  = normalPre.cosTheta( -localDirPre );
    }

    G4ThreeVector localPosPost  = transform.TransformPoint( globalPosPost );
    EInside       statusPost    = solid.Inside( localPosPost );
    if (statusPost == kSurface) {
      leave_sensitive = true;
      G4ThreeVector globalDirPost = step->GetPostStepPoint()->GetMomentumDirection();
      G4ThreeVector localDirPost  = transform.TransformAxis( globalDirPost );
      G4ThreeVector normalPost    = solid.SurfaceNormal( localPosPost );
      cosThetaPost = normalPost.cosTheta( localDirPost );
    }
  }

  // update track accounting
  if (enter_sensitive)
    m_track.enterDetector( sensitive, position, cosThetaPre );
  m_track.step(MaterialAccountingStep( length, radiationLengths, energyLoss, globalPositionIn, globalPositionOut ));
  if (leave_sensitive)
    m_track.leaveDetector( sensitive, cosThetaPost );

  if (sensitive)
    LogInfo("TrackingMaterialProducer") << "Track was near sensitive     volume "
                                         << sensitive->GetName() << std::endl;
  else
    LogInfo("TrackingMaterialProducer") << "Track was near non-sensitive volume "
                                         << touchable->GetVolume()->GetName() << std::endl;
  LogInfo("TrackingMaterialProducer")  << "Step length:             "
                                       << length << " cm\n"
                                       << "globalPreStep(r,z): (" << globalPositionIn.perp()
                                       << ", " << globalPositionIn.z() << ") cm\n"
                                       << "globalPostStep(r,z): (" << globalPositionOut.perp()
                                       << ", " << globalPositionOut.z() << ") cm\n"
                                       << "position(r,z): ("
                                       << position.perp()
                                       << ", " << position.z() << ") cm\n"
                                       << "Radiation lengths:       "
                                       << radiationLengths << " \t\t(X0: "
                                       << X0 << " cm)\n"
                                       << "Energy loss:             "
                                       << energyLoss << " MeV  \t(Xi: "
                                       << Xi << " MeV/cm)\n"
                                       << "Track was " << (enter_sensitive ? "entering " : "in none ")
                                       << "sensitive volume\n"
                                       << "Track was " << (leave_sensitive ? "leaving  " : "in none ")
                                       << "sensitive volume\n";

}


//-------------------------------------------------------------------------
void TrackingMaterialProducer::update(const EndOfTrack* event)
{
  const G4Track * track = (*event)();
  if (m_primaryTracks and track->GetParentID() != 0)
    return;

  radLen_vs_eta_->Fill(track->GetMomentum().eta(), m_track.summary().radiationLengths());
  m_tracks->push_back(m_track);

  // LogInfo
  LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: this track took "
                                       << m_track.steps().size()
                                       << " steps, and passed through "
                                       << m_track.detectors().size()
                                       << " sensitive detectors" << std::endl;
  LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: track length:       "
                                       << m_track.summary().length()
                                       << " cm" << std::endl;
  LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: radiation lengths: "
                                       << m_track.summary().radiationLengths() << std::endl;
  LogInfo("TrackingMaterialProducer") << "TrackingMaterialProducer: energy loss:        "
                                       << m_track.summary().energyLoss()
                                       << " MeV" << std::endl;
}

//-------------------------------------------------------------------------
void TrackingMaterialProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // transfer ownership to the Event
  std::unique_ptr<std::vector<MaterialAccountingTrack> > tracks( m_tracks );
  iEvent.put(std::move(tracks));
  m_tracks = nullptr;
}

//-------------------------------------------------------------------------
bool TrackingMaterialProducer::isSelected( const G4VTouchable* touchable )
{
  for (size_t i = 0; i < m_selectedVolumes.size(); ++i)
    if (m_selectedVolumes[i]->IsAncestor( touchable->GetVolume() )
        or m_selectedVolumes[i] == touchable->GetVolume()->GetLogicalVolume())
      return true;

  return false;
}

//-------------------------------------------------------------------------
// define as a plugin
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SIMWATCHER(TrackingMaterialProducer);
