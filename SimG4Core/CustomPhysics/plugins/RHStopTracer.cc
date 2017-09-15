#include "SimG4Core/CustomPhysics/interface/RHStopTracer.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
  
#include "G4Track.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

RHStopTracer::RHStopTracer(edm::ParameterSet const & p) {
  edm::ParameterSet parameters = p.getParameter<edm::ParameterSet>("RHStopTracer");
  mStopRegular = parameters.getUntrackedParameter<bool>("stopRegularParticles", false);
  mTraceEnergy = parameters.getUntrackedParameter<double>("traceEnergy", 1.e20); 
  mTraceParticleName = parameters.getParameter<std::string>("traceParticle");
  minPdgId = parameters.getUntrackedParameter<int>("minPdgId", 1000000); 
  maxPdgId = parameters.getUntrackedParameter<int>("maxPdgId", 2000000); 
  otherPdgId = parameters.getUntrackedParameter<int>("otherPdgId", 17); 
  produces< std::vector<std::string> >("StoppedParticlesName");
  produces< std::vector<float> >("StoppedParticlesX");
  produces< std::vector<float> >("StoppedParticlesY");
  produces< std::vector<float> >("StoppedParticlesZ");
  produces< std::vector<float> >("StoppedParticlesTime");
  produces< std::vector<int> >("StoppedParticlesPdgId");
  produces< std::vector<float> >("StoppedParticlesMass");
  produces< std::vector<float> >("StoppedParticlesCharge");

  mTraceEnergy *= CLHEP::GeV;
  rePartName = mTraceParticleName;

  edm::LogInfo("SimG4CoreCustomPhysics")
    << "RHStopTracer::RHStopTracer " << mTraceParticleName 
    << " Eth(GeV)= " << mTraceEnergy;

}

RHStopTracer::~RHStopTracer() {
}

void RHStopTracer::update (const BeginOfRun * fRun) {
  LogDebug("SimG4CoreCustomPhysics")
    << "RHStopTracer::update-> begin of the run " << (*fRun)()->GetRunID(); 
}

void RHStopTracer::update (const BeginOfEvent * fEvent) {
  LogDebug("SimG4CoreCustomPhysics")
    << "RHStopTracer::update-> begin of the event " << (*fEvent)()->GetEventID(); 
}

void RHStopTracer::update (const BeginOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  const G4ParticleDefinition* part = track->GetDefinition();
  const std::string& stringPartName = part->GetParticleName();
  bool matched = false;
  int pdgid = std::abs(part->GetPDGEncoding());
  if( (pdgid>minPdgId && pdgid<maxPdgId) || pdgid==otherPdgId )
     matched = std::regex_match(stringPartName,rePartName);
  if( matched ||  track->GetKineticEnergy() > mTraceEnergy) {
    LogDebug("SimG4CoreCustomPhysics")
      << "RHStopTracer::update-> new track: ID/Name/pdgId/mass/charge/Parent: " 
      << track->GetTrackID() << '/' << part->GetParticleName() << '/' 
      << part->GetPDGEncoding() << '/'
      << part->GetPDGMass()/GeV <<" GeV/" << part->GetPDGCharge() << '/'
      << track->GetParentID()
      << " Position: " << track->GetPosition() << ' ' 
      << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
      << "   4vec " << track->GetMomentum();
  } else if (mStopRegular) { // kill regular particles
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
  }
}

void RHStopTracer::update (const EndOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  const G4ParticleDefinition* part = track->GetDefinition();
  const std::string& stringPartName = part->GetParticleName();
  bool matched = false;
  int pdgid = std::abs(part->GetPDGEncoding());
  if( (pdgid>minPdgId && pdgid<maxPdgId) || pdgid==otherPdgId )
     matched = std::regex_match(stringPartName,rePartName);
  if( matched ||  track->GetKineticEnergy() > mTraceEnergy) {
    LogDebug("SimG4CoreCustomPhysics")
      << "RHStopTracer::update-> stop track: ID/Name/pdgId/mass/charge/Parent: " 
      << track->GetTrackID() << '/' << part->GetParticleName() << '/' 
      << part->GetPDGEncoding() << '/'
      << part->GetPDGMass()/GeV <<" GeV/" << part->GetPDGCharge() << '/'
      << track->GetParentID()
      << " Position: " << track->GetPosition() << ' ' 
      << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
      << "   4vec " << track->GetMomentum();
    if (track->GetMomentum().mag () < 0.001) {
      LogDebug("SimG4CoreCustomPhysics") <<
	"RHStopTracer:: track has stopped, so making StopPoint";
      mStopPoints.push_back (StopPoint (track->GetDefinition()->GetParticleName(),
					track->GetPosition().x(),
					track->GetPosition().y(),
					track->GetPosition().z(),
					track->GetGlobalTime(),
					track->GetDefinition()->GetPDGEncoding(),
                                        track->GetDefinition()->GetPDGMass()/GeV,
                                        track->GetDefinition()->GetPDGCharge() ));
    }
  }
}

void RHStopTracer::produce(edm::Event& fEvent, const edm::EventSetup&) {
  LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::produce->";

   std::unique_ptr<std::vector<std::string> > names(new std::vector<std::string>); 
   std::unique_ptr<std::vector<float> > xs(new std::vector<float>);
   std::unique_ptr<std::vector<float> > ys(new std::vector<float>);
   std::unique_ptr<std::vector<float> > zs(new std::vector<float>);
   std::unique_ptr<std::vector<float> > ts(new std::vector<float>);
   std::unique_ptr<std::vector<int> > ids(new std::vector<int>);
   std::unique_ptr<std::vector<float> > masses(new std::vector<float>);
   std::unique_ptr<std::vector<float> > charges(new std::vector<float>);

   std::vector <StopPoint>::const_iterator stopPoint = mStopPoints.begin ();
   for (;  stopPoint != mStopPoints.end(); ++stopPoint) {
     names->push_back (stopPoint->name);
     xs->push_back (stopPoint->x);
     ys->push_back (stopPoint->y);
     zs->push_back (stopPoint->z);
     ts->push_back (stopPoint->t);
     ids->push_back (stopPoint->id);
     masses->push_back (stopPoint->mass);
     charges->push_back (stopPoint->charge);
   }
   fEvent.put(std::move(names), "StoppedParticlesName");
   fEvent.put(std::move(xs), "StoppedParticlesX");
   fEvent.put(std::move(ys), "StoppedParticlesY");
   fEvent.put(std::move(zs), "StoppedParticlesZ");
   fEvent.put(std::move(ts), "StoppedParticlesTime");
   fEvent.put(std::move(ids), "StoppedParticlesPdgId");
   fEvent.put(std::move(masses), "StoppedParticlesMass");
   fEvent.put(std::move(charges), "StoppedParticlesCharge");
   mStopPoints.clear ();
}
