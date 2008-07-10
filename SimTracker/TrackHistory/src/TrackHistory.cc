/*
 *  TrackHistory.cc
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 7/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"

TrackHistory::TrackHistory (
  const edm::ParameterSet & config
)
{
  // Default depth	
  depth_ = -1;
			
  // Name of the track collection
  trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );

  // Name of the traking pariticle collection
  trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );
  
  // Track association record
  trackAssociator_ = config.getUntrackedParameter<std::string> ( "trackAssociator" );

  // Association by max. value
  bestMatchByMaxValue_ = config.getUntrackedParameter<bool> ( "bestMatchByMaxValue" );
}


void TrackHistory::newEvent (
  const edm::Event & event, const edm::EventSetup & setup
)
{
  // Track collection
  edm::Handle<edm::View<reco::Track> > trackCollection;
  event.getByLabel(trackProducer_, trackCollection);
   
  // Tracking particle information
  edm::Handle<TrackingParticleCollection>  TPCollection;
  event.getByLabel(trackingTruth_, TPCollection);
    
  // Get the track associator
  edm::ESHandle<TrackAssociatorBase> associator;
  setup.get<TrackAssociatorRecord>().get(trackAssociator_, associator);
  
  // Get the map between recotracks and tp
  association_ = associator->associateRecoToSim (trackCollection, TPCollection, &event);
}


void TrackHistory::traceGenHistory(const HepMC::GenParticle * gpp)
{
  // Third stop criteria: status abs(depth_) particles after the hadronization.
  // The after hadronization is done by detecting the pdg_id pythia code from 88 to 99
  if ( gpp->status() <= abs(depth_) && (gpp->pdg_id() < 88 || gpp->pdg_id() > 99) )
  {
    genParticleTrail_.push_back(gpp);
    // Get the producer vertex.
    HepMC::GenVertex * vertex = gpp->production_vertex();
    // Verify if has a vertex associated
    if ( vertex )
    {
      genVertexTrail_.push_back(vertex);
      if ( vertex->particles_in_size()  ) // Verify if the vertex has incoming particles
        traceGenHistory( *(vertex->particles_in_const_begin()) );
    }
  } 
}


bool TrackHistory::traceSimHistory(TrackingParticleRef tpr, int depth)
{
  // first stop condition: if the required depth is reached
  if ( depth == depth_ && depth_ >= 0 ) return true;

  // sencond stop condition: if a gen particle is associated to the TP  
  if ( !tpr->genParticle().empty() ) 
  {
    LogDebug("TrackHistory") << "Particle " << tpr->pdgId() << " has a GenParicle image." << std::endl;      
    traceGenHistory(&(**(tpr->genParticle_begin())));
    return true;
  }  
   
  // get a reference to the TP's parent vertex
  TrackingVertexRef parentVertex = tpr->parentVertex();
  
  // verify if the parent vertex exists  
  if ( parentVertex.isNonnull() ) 
  {
    // save the vertex in the trail
    simVertexTrail_.push_back(parentVertex);
    
    if ( !parentVertex->sourceTracks().empty() ) 
    {
      LogDebug("TrackHistory") << "No GenParticle image for " << tpr->pdgId() << " moving on to the parent particle." << std::endl;
          
      // select the original source in case of combined vertexes
      bool flag = false;
      TrackingVertex::tp_iterator itd, its;
      
      for(its = parentVertex->sourceTracks_begin(); its != parentVertex->sourceTracks_end(); its++) 
      {
        for(itd = parentVertex->daughterTracks_begin(); itd != parentVertex->daughterTracks_end(); itd++)
          if (itd != its) 
          {
            flag = true;
            break;
          }
        if (flag)
          break;
      }
      
      // verify if the new particle is not in the trail (looping partiles)
      if (
        std::find(
          simParticleTrail_.begin(),
          simParticleTrail_.end(),
          *its
        ) != simParticleTrail_.end()
      ) 
      {
        LogDebug("TrackHistory") <<  "WARNING: Looping track found." << std::endl;
        return false;
      } 
      
      // save particle in the trail
      simParticleTrail_.push_back(*its);
      return traceSimHistory (*its, --depth);
    }
    else 
    {
      LogDebug("TrackHistory") <<  "WARNING: Source track for tracking vertex cannot be found." << std::endl;
    }
  }
  else
  {
    LogDebug("TrackHistory") << " WARNING: Parent vertex for tracking particle cannot be found.";
  }
    
  return false;
}


bool TrackHistory::evaluate ( edm::RefToBase<reco::Track> tr )
{
  TrackingParticleRef tpr( match(tr) );

  if ( !tpr.isNull() )
  {
    evaluate(tpr);
    return true;
  }
  
  return false;
}


TrackingParticleRef TrackHistory::match ( edm::RefToBase<reco::Track> tr )
{
  TrackingParticleRef tpr;
  std::vector<std::pair<TrackingParticleRef, double> > tp;

  try
  {	
    tp = association_[tr];
  }
  catch (edm::Exception event) 
  {
  	return tpr;
  }

  double m = 0;
    
  for (std::size_t i=0; i<tp.size(); i++) 
  {
    if ( bestMatchByMaxValue_ ) 
    {
      if (i && tp[i].second > m) 
      {
        tpr = tp[i].first;
        m = tp[i].second;
      }
      else 
      {
        tpr = tp[i].first;
        m = tp[i].second;
      }
    } 
    else 
    {
      if (i && tp[i].second < m) 
      {
        tpr = tp[i].first;
        m = tp[i].second;
      }
      else
      {
        tpr = tp[i].first;
        m = tp[i].second;
      }
    }
  }

  return tpr;
}

