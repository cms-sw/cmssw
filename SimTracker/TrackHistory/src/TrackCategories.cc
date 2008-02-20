/*
 *  TrackCategories.C
 */

#include <math.h>

#include "HepPDT/ParticleID.hh"// user include files
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"#include "SimTracker/TrackHistory/interface/TrackCategories.h"	


TrackCategories::TrackCategories (
  const edm::ParameterSet & iConfig
)
{
  // Initialize flags	
  reset();

  // Set the history depth after hadronization
  tracer_.depth(-2);
		
  // Name of the track collection 
  trackCollection_ = iConfig.getParameter<std::string> ( "trackCollection" );

  // Association by hit
  associationByHits_ = iConfig.getParameter<bool> ( "associationByHits" );
}


void TrackCategories::event (
  const edm::Event & iEvent, const edm::EventSetup & iSetup
)
{
  // Track collection  edm::Handle<edm::View<reco::Track> > trackCollection;
  iEvent.getByLabel(trackCollection_, trackCollection);
  
  // Tracking particle information  edm::Handle<TrackingParticleCollection>  TPCollection;
  iEvent.getByType(TPCollection);

  // Magnetic field  iSetup.get<IdealMagneticFieldRecord>().get(magneticField_);
  
  // Trasient track builder  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", transientTrackBuilder_);

  // Get the associator by hits or chi2  edm::ESHandle<TrackAssociatorBase> associator;  if(associationByHits_)    iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", associator);  else      iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2", associator);
    
  association_ = associator->associateRecoToSim (trackCollection, TPCollection, &iEvent);
}


bool TrackCategories::evaluate (edm::RefToBase<reco::Track> track)
{
  // Reset all the flags;	
  reset();
    
  // Check if the track is a fake
  if ( tracer_.evaluate(track, association_, associationByHits_) )
  {
  	// Set fake flag
    flags_[Fake] = true; 
    
    // Classify by reconstructed information
    byReco(track);
    
    // Classify by track history information
    byHistory();
  }
  else
    flags_[Fake] = false;
    
  return flags_[Unknown];
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
  
  return flags_[Unknown];
}


void TrackCategories::byHistory()
{
	
  // Check for the existence of a simulated vertex (displaced).
  if ( tracer_.simVertexTrail().empty() )
    flags_[Displaced] = false;
  else
    flags_[Displaced] = true;
		
  // Get the simulated particle.
  const HepMC::GenParticle * particle = tracer_.particle();
    
  if (particle)
  {
    HepPDT::ParticleID pid(particle->pdg_id());
    flags_[Up] = pid.hasUp();
    flags_[Down] = pid.hasDown();
    flags_[Strange] = pid.hasStrange();
    flags_[Charm] = pid.hasCharm();
    flags_[Bottom] = pid.hasBottom();
    flags_[Light] = pid.hasUp() || pid.hasDown() || pid.hasStrange(); 
  }
  else
    flags_[Unknown] = true;
}


void TrackCategories::byReco(edm::RefToBase<reco::Track> track)
{
  // Find a TrackingParticle for the given RecoTrack 
  std::vector<std::pair<TrackingParticleRef, double> > tp;
     
  try  {    tp = association_[track];  }  catch (edm::Exception e) {}

  // Get track with maximum match.  double match = 0;  TrackingParticleRef tpr;
  
  for (std::size_t i=0; i<tp.size(); i++)   {    if (associationByHits_)     {      if (i && tp[i].second > match)       {        tpr = tp[i].first;        match = tp[i].second;      }      else       {        tpr = tp[i].first;        match = tp[i].second;      }    }     else     {      if (i && tp[i].second < match)       {        tpr = tp[i].first;        match = tp[i].second;      }      else      {        tpr = tp[i].first;        match = tp[i].second;      }    }  }
  // Compute tracking particle parameters at point of closest approach to the beamline  const SimTrack * assocTrack = &(*tpr->g4Track_begin());   FreeTrajectoryState ftsAtProduction(    GlobalPoint(      tpr->vertex().x(),      tpr->vertex().y(),      tpr->vertex().z()    ),    GlobalVector(      assocTrack->momentum().x(),      assocTrack->momentum().y(),      assocTrack->momentum().z()    ),     TrackCharge(track->charge()),    magneticField_.product()  );        TSCPBuilderNoMaterial tscpBuilder;    TrajectoryStateClosestToPoint tsAtClosestApproach = tscpBuilder(    ftsAtProduction,    GlobalPoint(0,0,0)  );    GlobalPoint v = tsAtClosestApproach.theState().position();  GlobalVector p = tsAtClosestApproach.theState().momentum();   
  // Simulated d0  double d0Sim = - (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
  
  // Calculate the d0 pull
  double d0Pull = ( track->d0() - d0Sim ) / track->d0Error();
  
  // Return true if d0Pull < 3 sigmas
  flags_[Bad] = (d0Pull < 3.0) ? false: true;  
}


