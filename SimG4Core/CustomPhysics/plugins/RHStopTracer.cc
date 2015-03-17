#include "RHStopTracer.h"

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

RHStopTracer::RHStopTracer(edm::ParameterSet const & p) {
  edm::ParameterSet parameters = p.getParameter<edm::ParameterSet>("RHStopTracer");
  mStopRegular = parameters.getUntrackedParameter<bool>("stopRegularParticles", false);
  mTraceEnergy = 1000 * parameters.getUntrackedParameter<double>("traceEnergy", 1.e20); // GeV->KeV
  mTraceParticleNameRegex = parameters.getParameter<std::string>("traceParticle");
  produces< std::vector<std::string> >("StoppedParticlesName");
  produces< std::vector<float> >("StoppedParticlesX");
  produces< std::vector<float> >("StoppedParticlesY");
  produces< std::vector<float> >("StoppedParticlesZ");
  produces< std::vector<float> >("StoppedParticlesTime");
  produces< std::vector<int> >("StoppedParticlesPdgId");
  produces< std::vector<float> >("StoppedParticlesMass");
  produces< std::vector<float> >("StoppedParticlesCharge");

  LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::RHStopTracer->" 
				     << mTraceParticleNameRegex << '/' << mTraceEnergy;
}

RHStopTracer::~RHStopTracer() {
}

void RHStopTracer::update (const BeginOfRun * fRun) {
  LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::update-> begin of the run " << (*fRun)()->GetRunID(); 
}

void RHStopTracer::update (const BeginOfEvent * fEvent) {
  LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::update-> begin of the event " << (*fEvent)()->GetEventID(); 
}

void RHStopTracer::update (const BeginOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  if ((track->GetMomentum().mag()> mTraceEnergy) || matched (track->GetDefinition()->GetParticleName())) {
    LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::update-> new track: ID/Name/pdgId/mass/charge/Parent: " 
				       << track->GetTrackID() << '/' << track->GetDefinition()->GetParticleName() << '/' 
				       << track->GetDefinition()->GetPDGEncoding() << '/'
				       << track->GetDefinition()->GetPDGMass()/GeV <<" GeV/" << track->GetDefinition()->GetPDGCharge() << '/'
				       << track->GetParentID()
				       << " position X/Y/Z: " << track->GetPosition().x() << '/' 
				       << track->GetPosition().y() << '/' <<  track->GetPosition().z()
				       << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
				       << "    px/py/pz/p=" << track->GetMomentum().x() << '/' 
				       << track->GetMomentum().y() << '/' << track->GetMomentum().z() << '/'<< track->GetMomentum().mag(); 
  }
  if (mStopRegular && !matched (track->GetDefinition()->GetParticleName())) { // kill regular particles
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
  }
}

void RHStopTracer::update (const EndOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  if ((track->GetMomentum().mag()> mTraceEnergy) || matched (track->GetDefinition()->GetParticleName())) {
    LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::update-> stop track: ID/Name/pdgId/mass/charge/Parent: " 
				       << track->GetTrackID() << '/' << track->GetDefinition()->GetParticleName() << '/' 
				       << track->GetDefinition()->GetPDGEncoding() << '/'
				       << track->GetDefinition()->GetPDGMass()/GeV <<" GeV/" << track->GetDefinition()->GetPDGCharge() << '/'
				       << track->GetParentID()
				       << " position X/Y/Z: " << track->GetPosition().x() << '/' 
				       << track->GetPosition().y() << '/' <<  track->GetPosition().z()
				       << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
				       << "    px/py/pz/p=" << track->GetMomentum().x() << '/' 
				       << track->GetMomentum().y() << '/' << track->GetMomentum().z() << '/'<< track->GetMomentum().mag(); 
    if (track->GetMomentum().mag () < 0.001) {
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

bool RHStopTracer::matched (const std::string& fName) const {
  return boost::regex_match (fName, mTraceParticleNameRegex);
}

 void RHStopTracer::produce(edm::Event& fEvent, const edm::EventSetup&) {
   LogDebug("SimG4CoreCustomPhysics") << "RHStopTracer::produce->";

   std::auto_ptr<std::vector<std::string> > names (new std::vector<std::string>); 
   std::auto_ptr<std::vector<float> > xs (new std::vector<float>);
   std::auto_ptr<std::vector<float> > ys (new std::vector<float>);
   std::auto_ptr<std::vector<float> > zs (new std::vector<float>);
   std::auto_ptr<std::vector<float> > ts (new std::vector<float>);
   std::auto_ptr<std::vector<int> > ids (new std::vector<int>);
   std::auto_ptr<std::vector<float> > masses (new std::vector<float>);
   std::auto_ptr<std::vector<float> > charges (new std::vector<float>);

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
   fEvent.put (names, "StoppedParticlesName");
   fEvent.put (xs, "StoppedParticlesX");
   fEvent.put (ys, "StoppedParticlesY");
   fEvent.put (zs, "StoppedParticlesZ");
   fEvent.put (ts, "StoppedParticlesTime");
   fEvent.put (ids, "StoppedParticlesPdgId");
   fEvent.put (masses, "StoppedParticlesMass");
   fEvent.put (charges, "StoppedParticlesCharge");
   mStopPoints.clear ();
 }
