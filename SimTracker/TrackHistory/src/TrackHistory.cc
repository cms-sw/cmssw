/*
 *  TrackHistory.cc
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 7/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */


#include "SimTracker/TrackHistory/interface/TrackHistory.h"	


void TrackHistory::traceGenHistory(const HepMC::GenParticle * gpp)
{
  // Third stop criteria: status abs(depth_) particles after the hadronization.
  if ( gpp->status() <= abs(depth_) && gpp->pdg_id() != 92 )
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
    std::cout << "Particle " << tpr->pdgId() << " has a GenParicle image." << std::endl;
    for (TrackingParticle::genp_iterator hepT = tpr->genParticle_begin(); hepT !=  tpr->genParticle_end(); ++hepT)
      std::cout << "  HepMC Momentum :" << (*hepT)->momentum().mag() << (*hepT)->pdg_id() << std::endl;
      
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
      std::cout << "No GenParticle image for " << tpr->pdgId() << " moving on to the parent particle." << std::endl;
      
      TrackingVertex::tp_iterator it = parentVertex->daughterTracks_begin();
      
      for(; it != parentVertex->daughterTracks_end(); it++)
        std::cout << "  Vertex daughter " << (*it)->pdgId() << " moment: " << (*it)->pt() << " matched hits " << (*it)->matchedHit() << std::endl;
      
      it = parentVertex->sourceTracks_begin();
      
      for(; it != parentVertex->sourceTracks_end(); it++)
        std::cout << "  Vertex source " << (*it)->pdgId() << " moment: " << (*it)->pt() << " matched hits " << (*it)->matchedHit() << std::endl;
    
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
        std::cout <<  "WARNING: Looping track found." << std::endl;
        return false;
      } 
      
      // save particle in the trail
      simParticleTrail_.push_back(*its);
      return traceSimHistory (*its, --depth);
    }
    else 
    {
      std::cout <<  "WARNING: Source track for tracking vertex cannot be found." << std::endl;
    }
  }
  else
  {
    std::cout << " WARNING: Parent vertex for tracking particle cannot be found.";
  }
    
  return false;
}


bool TrackHistory::evaluate (
  reco::TrackRef tr,
  reco::RecoToSimCollection const & association, 
  bool maxMatch
)
{
  try
  {
    std::vector<std::pair<TrackingParticleRef, double> > tp = association[tr];
    
    // get the track with maximum match
    double match = 0;
    TrackingParticleRef tpr;
    
    for (std::size_t i=0; i<tp.size(); i++) 
    {
      if (maxMatch) 
      {
        if (i && tp[i].second > match) 
        {
          tpr = tp[i].first;
          match = tp[i].second;
        }
        else 
        {
          tpr = tp[i].first;
          match = tp[i].second;
        }
      } 
      else 
      {
        if (i && tp[i].second < match) 
        {
          tpr = tp[i].first;
          match = tp[i].second;
        }
        else
        {
          tpr = tp[i].first;
          match = tp[i].second;
        }
      }
    }
    // evaluate history for the best match.
    evaluate(tpr);
    return true;
  }
  catch (edm::Exception event) {}
  return false;
}


