#include "RHStopTracer.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "FWCore/Framework/interface/Event.h"
  
#include "G4Track.hh"
#include "G4Run.hh"
#include "G4Event.hh"


RHStopTracer::RHStopTracer(edm::ParameterSet const & p) {
  edm::ParameterSet parameters = p.getParameter<edm::ParameterSet>("RHStopTracer");
  mDebug = parameters.getUntrackedParameter<bool>("verbose", false);
  mStopRegular = parameters.getUntrackedParameter<bool>("stopRegularParticles", false);
  mTraceEnergy = 1000 * parameters.getUntrackedParameter<double>("traceEnergy", 1.e20); // GeV->KeV
  mTraceParticleNameRegex = parameters.getParameter<std::string>("traceParticle");
  produces< std::vector<std::string> >("StoppedParticlesName");
  produces< std::vector<float> >("StoppedParticlesX");
  produces< std::vector<float> >("StoppedParticlesY");
  produces< std::vector<float> >("StoppedParticlesZ");
  produces< std::vector<float> >("StoppedParticlesTime");

  if (mDebug) {
    std::cout << "RHStopTracer::RHStopTracer->" 
	      << mTraceParticleNameRegex << '/' << mTraceEnergy << std::endl;
  }
}

RHStopTracer::~RHStopTracer() {
}

void RHStopTracer::update (const BeginOfRun * fRun) {
  if (mDebug) 
    std::cout << "RHStopTracer::update-> begin of the run " << (*fRun)()->GetRunID () << std::endl; 
}

void RHStopTracer::update (const BeginOfEvent * fEvent) {
  if (mDebug) 
    std::cout << "RHStopTracer::update-> begin of the event " << (*fEvent)()->GetEventID () << std::endl; 
}

void RHStopTracer::update (const BeginOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  if ((track->GetMomentum().mag()> mTraceEnergy) || matched (track->GetDefinition()->GetParticleName())) {
    if (mDebug)
    std::cout << "RHStopTracer::update-> new track: ID/Name/mass/Parent: " 
	      << track->GetTrackID() << '/' << track->GetDefinition()->GetParticleName() << '/' 
	      << track->GetDefinition()->GetPDGMass() << '/' << track->GetParentID()
	      << std::endl
	      << " position X/Y/Z: " << track->GetPosition().x() << '/' 
	      << track->GetPosition().y() << '/' <<  track->GetPosition().z()
	      << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
	      << std::endl
	      << "    px/py/pz/p=" << track->GetMomentum().x() << '/' 
	      << track->GetMomentum().y() << '/' << track->GetMomentum().z() << '/'<< track->GetMomentum().mag() 
	      << std::endl;
  }
  if (mStopRegular && !matched (track->GetDefinition()->GetParticleName())) { // kill regular particles
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
  }
}

void RHStopTracer::update (const EndOfTrack * fTrack) {
  const G4Track* track = (*fTrack)();
  if ((track->GetMomentum().mag()> mTraceEnergy) || matched (track->GetDefinition()->GetParticleName())) {
    if (mDebug)
    std::cout << "RHStopTracer::update-> stop track: ID/Name/mass/Parent: " 
	      << track->GetTrackID() << '/' << track->GetDefinition()->GetParticleName() << '/' 
	      << track->GetDefinition()->GetPDGMass() << '/' << track->GetParentID()
	      << std::endl
	      << " position X/Y/Z: " << track->GetPosition().x() << '/' 
	      << track->GetPosition().y() << '/' <<  track->GetPosition().z()
	      << " R/phi: " << track->GetPosition().perp() << '/' << track->GetPosition().phi()
	      << std::endl 
	      << "    px/py/pz/p=" << track->GetMomentum().x() << '/' 
	      << track->GetMomentum().y() << '/' << track->GetMomentum().z() << '/'<< track->GetMomentum().mag() 
	      << std::endl;
    if (track->GetMomentum().mag () < 0.001) {
      mStopPoints.push_back (StopPoint (track->GetDefinition()->GetParticleName(),
					track->GetPosition().x(),
					track->GetPosition().y(),
					track->GetPosition().z(),
					track->GetGlobalTime()));
    }
  }
}

bool RHStopTracer::matched (const std::string& fName) const {
  return boost::regex_match (fName, mTraceParticleNameRegex);
}

 void RHStopTracer::produce(edm::Event& fEvent, const edm::EventSetup&) {
  if (mDebug) {
    std::cout << "RHStopTracer::produce->" << std::endl;
  }
   std::auto_ptr<std::vector<std::string> > names (new std::vector<std::string>); 
   std::auto_ptr<std::vector<float> > xs (new std::vector<float>);
   std::auto_ptr<std::vector<float> > ys (new std::vector<float>);
   std::auto_ptr<std::vector<float> > zs (new std::vector<float>);
   std::auto_ptr<std::vector<float> > ts (new std::vector<float>);

   std::vector <StopPoint>::const_iterator stopPoint = mStopPoints.begin ();
   for (;  stopPoint != mStopPoints.end(); ++stopPoint) {
     names->push_back (stopPoint->name);
     xs->push_back (stopPoint->x);
     ys->push_back (stopPoint->y);
     zs->push_back (stopPoint->z);
     ts->push_back (stopPoint->t);
   }
   fEvent.put (names, "StoppedParticlesName");
   fEvent.put (xs, "StoppedParticlesX");
   fEvent.put (ys, "StoppedParticlesY");
   fEvent.put (zs, "StoppedParticlesZ");
   fEvent.put (ts, "StoppedParticlesTime");
   mStopPoints.clear ();
 }
