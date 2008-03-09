/*
 *  TrackCategories.C
 */

#include <math.h>
#include <iostream>

#include "HepPDT/ParticleID.hh"

// user include files
#include "SimTracker/TrackHistory/interface/TrackCategories.h"	


void TrackCategories::newEvent (
  const edm::Event & iEvent, const edm::EventSetup & iSetup
)
{
  // Get the new event information for the tracer	
  tracer_.newEvent(iEvent, iSetup);
  
  // Magnetic field
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField_);
  
  // Trasient track builder
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", transientTrackBuilder_);
}


bool TrackCategories::evaluate (edm::RefToBase<reco::Track> track)
{
  // Reset all the flags;	
  reset();
    
  // Check if the track is a fake
  if ( tracer_.evaluate(track) )
  {  	
  	// Set fake flag
    flags_[Fake] = false; 
    
    // Classify by reconstructed information
    byReco(track);
    
    // Classify by track history information
    byHistory();
  }
  else
    flags_[Fake] = true;
    
  return !flags_[Unknown];
}


bool TrackCategories::evaluate (TrackingParticleRef track)
{
  // Reset all the flags;	
  reset();
    
  // Check if the track history can be traced
  if ( tracer_.evaluate(track) )
  {
    // Classify by track history information
    byHistory();
  }
  else
    flags_[Unknown] = true;
  
  return !flags_[Unknown];
}


void TrackCategories::byHistory()
{
  // Get the event id for the initial TP.
  EncodedEventId eventId = tracer_.simParticle()->eventId();
  
  // Check for signal events.	
  if ( !eventId.bunchCrossing() && !eventId.event() )
  {
    flags_[SignalEvent] = true;
    // Check for PV, SV, TV
    TrackHistory::GenVertexTrail genVertexTrail(tracer_.genVertexTrail());
    if ( genVertexTrail.empty() )
      flags_[PV] = true;
    else if ( genVertexTrail.size() == 1 )
      flags_[SV] = true;
    else
      flags_[TV] = true;
  }
 	
  // Check for the existence of a simulated vertex (displaced).
  if ( !tracer_.simVertexTrail().empty() )
    flags_[Displaced] = true;
		
  // Checks for long lived particle
  flags_[Ks] = hasLongLived(310);      // Ks
  flags_[Lambda] = hasLongLived(3122); // Lambda 
  
  // Check for photon conversion
  flags_[PhotonConversion] = hasPhotonConversion();
	
  // Get the simulated particle.
  const HepMC::GenParticle * particle = tracer_.particle();
  
  // Check for the initial hadron
  if (particle)
  {
    HepPDT::ParticleID pid(particle->pdg_id());
    flags_[Up] = pid.hasUp();
    flags_[Down] = pid.hasDown();
    flags_[Strange] = pid.hasStrange();
    flags_[Charm] = pid.hasCharm();
    flags_[Bottom] = pid.hasBottom();
    flags_[Light] = !pid.hasCharm() || !pid.hasBottom();
  }
  else
    flags_[Unknown] = true;
}

bool TrackCategories::hasLongLived(int pdgid) const
{
  if ( !tracer_.genParticleTrail().empty() )
  {
    if( abs(tracer_.genParticleTrail()[0]->pdg_id()) == pdgid )
      return true;
  }
  return false;}

bool TrackCategories::hasPhotonConversion() const{  TrackOrigin::SimVertexTrail::const_iterator tvr;  TrackingParticleRefVector sources, daughters;  for (tvr = tracer_.simVertexTrail().begin(); tvr != tracer_.simVertexTrail().end(); tvr++)  {    sources = (*tvr)->sourceTracks();    daughters = (*tvr)->daughterTracks();    if (sources.size() == 1              &&  // require one source         daughters.size() == 2            &&  //    "    two daughters        sources[0]->pdgId() == 22        &&  //    "    a photon in the source        abs(daughters[0]->pdgId()) == 11 &&  //    "    two electrons        abs(daughters[1]->pdgId()) == 11    ) return true;  }  return false;}

void TrackCategories::byReco(edm::RefToBase<reco::Track> track)
{
  TrackingParticleRef tpr = tracer_.simParticle();

  // Compute tracking particle parameters at point of closest approach to the beamline
  const SimTrack * assocTrack = &(*tpr->g4Track_begin());
 
  FreeTrajectoryState ftsAtProduction(
    GlobalPoint(
      tpr->vertex().x(),
      tpr->vertex().y(),
      tpr->vertex().z()
    ),
    GlobalVector(
      assocTrack->momentum().x(),
      assocTrack->momentum().y(),
      assocTrack->momentum().z()
    ), 
    TrackCharge(track->charge()),
    magneticField_.product()
  );
      
  TSCPBuilderNoMaterial tscpBuilder;
  
  TrajectoryStateClosestToPoint tsAtClosestApproach = tscpBuilder(
    ftsAtProduction,
    GlobalPoint(0,0,0)
  );
  
  GlobalPoint v = tsAtClosestApproach.theState().position();
  GlobalVector p = tsAtClosestApproach.theState().momentum(); 
  
  // Simulated d0
  double d0Sim = - (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
  
  // Calculate the d0 pull
  double d0Pull = ( track->d0() - d0Sim ) / track->d0Error();
  
  // Return true if d0Pull < 3 sigmas
  flags_[Bad] = (d0Pull < 3.0) ? false: true;  
}


