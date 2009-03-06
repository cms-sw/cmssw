/*
 *  SimpleTHA.C
 *
 *  Created by Victor Eduardo Bazterra on 5/31/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */
 
// system include files
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

// user include files

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimTracker/TrackHistory/interface/TrackOrigin.h"

//
// class decleration
//

class SimpleTHA : public edm::EDAnalyzer 
{
public:

  explicit SimpleTHA(const edm::ParameterSet&);
  ~SimpleTHA();

private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  // Member data
  
  std::string trackCollection_;
      
  std::size_t totalTracks_;
   
  edm::ESHandle<ParticleDataTable> pdt_;
   
  std::string particleString(int) const;
  
  TrackHistory tracer_;
  
  std::string vertexString(
    TrackingParticleRefVector,
    TrackingParticleRefVector
  ) const;
   
  std::string vertexString(
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator
  ) const;
};


SimpleTHA::SimpleTHA(const edm::ParameterSet& iConfig) : tracer_(iConfig)
{
  trackCollection_ = iConfig.getParameter<std::string> ( "recoTrackModule" );
}


SimpleTHA::~SimpleTHA() { }


void SimpleTHA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Track collection
  edm::Handle<edm::View<reco::Track> > trackCollection;
  iEvent.getByLabel(trackCollection_,trackCollection);

  tracer_.depth(-2);
  tracer_.newEvent(iEvent, iSetup);
  
  // Loop over the track collection.
  for (std::size_t index = 0; index < trackCollection->size(); index++)
  {

    // If the track is not fake then get the vertexes
    if ( tracer_.evaluate( edm::RefToBase<reco::Track>(trackCollection, index) ) )
    {
      // Get the list of TrackingParticles associated to
      TrackOrigin::SimParticleTrail simParticles(tracer_.simParticleTrail());

      std::cout << std::endl << "History for track #" << index << " : " << std::endl;

      // Loop over all simParticles
      for(std::size_t hindex=0; hindex<simParticles.size(); hindex++) 
      {
        std::cout << "  simParticles [" << hindex << "] : " 
                  << particleString(simParticles[hindex]->pdgId()) 
                  << std::endl;
      }
      
      // Get the list of TrackingVertexes associated to
      TrackOrigin::SimVertexTrail simVertexes(tracer_.simVertexTrail());
         
      // Loop over all simVertexes                       
      if( !simVertexes.empty() )
      {
        for(std::size_t hindex=0; hindex<simVertexes.size(); hindex++) 
        {
          std::cout << "  simVertex    [" << hindex << "] : "
                    << vertexString(
                         simVertexes[hindex]->sourceTracks(),
                         simVertexes[hindex]->daughterTracks()                      
                       ) 
                    << std::endl;
        }
      }
      else
        std::cout << "  simVertex no found" << std::endl;

      // Get the list of GenParticles associated to
      TrackHistory::GenParticleTrail genParticles(tracer_.genParticleTrail());

      // Loop over all genParticles
      for(std::size_t hindex=0; hindex<genParticles.size(); hindex++) 
      {
        std::cout << "  genParticles [" << hindex << "] : " 
                  << particleString(genParticles[hindex]->pdg_id()) 
                  << std::endl;
      }
      
      // Get the list of TrackingVertexes associated to
      TrackHistory::GenVertexTrail genVertexes(tracer_.genVertexTrail());
         
      // Loop over all simVertexes                       
      if( !genVertexes.empty() )
      {
        for(std::size_t hindex=0; hindex<genVertexes.size(); hindex++) 
        {
          std::cout << "  genVertex    [" << hindex << "] : "
                    << vertexString(
                         genVertexes[hindex]->particles_in_const_begin(),
                         genVertexes[hindex]->particles_in_const_end(),
                         genVertexes[hindex]->particles_out_const_begin(),
                         genVertexes[hindex]->particles_out_const_end()                                               
                       ) 
                    << std::endl;
        }
      }
      else
        std::cout << "  genVertex no found" << std::endl;
    }
    else
      std::cout << "  Fake track" << std::endl;

    std::cout << std::endl;
  }
}


void 
SimpleTHA::beginJob(const edm::EventSetup& iSetup) 
{
  // Get the particles table.
  iSetup.getData( pdt_ );
  
  totalTracks_ = 0;
}


std::string SimpleTHA::particleString(int pdgId) const
{
  ParticleData const * pid;

  std::ostringstream vDescription;
    
  HepPDT::ParticleID particleType(pdgId);

  if (particleType.isValid())
  {
     pid = pdt_->particle(particleType);
     if (pid)
       vDescription << pid->name();
     else
       vDescription << pdgId;            
  }
  else
    vDescription << pdgId;

  return vDescription.str();
}


std::string SimpleTHA::vertexString(
  TrackingParticleRefVector in,
  TrackingParticleRefVector out
) const
{
  ParticleData const * pid;

  std::ostringstream vDescription;
    
  for(std::size_t j = 0; j < in.size(); j++)
  {
    if (!j) vDescription << "(";

    HepPDT::ParticleID particleType(in[j]->pdgId());

    if (particleType.isValid())
    {
       pid = pdt_->particle(particleType);
       if (pid)
         vDescription << pid->name();
       else
         vDescription << in[j]->pdgId();            
    }
    else
      vDescription << in[j]->pdgId();

    if (j == in.size() - 1) vDescription << ")";
    else vDescription << ",";
  }
        
  vDescription << "->";
        
  for(std::size_t j = 0; j < out.size(); j++)
  {
    if (!j) vDescription << "(";
          
    HepPDT::ParticleID particleType(out[j]->pdgId());

    if (particleType.isValid())
    {
      pid = pdt_->particle(particleType);
      if (pid)
        vDescription << pid->name();
      else
        vDescription << out[j]->pdgId();            
    }
    else
      vDescription << out[j]->pdgId();
      
    if (j == out.size() - 1) vDescription << ")";
    else vDescription << ",";
  }

  return vDescription.str();
}


std::string SimpleTHA::vertexString(
  HepMC::GenVertex::particles_in_const_iterator in_begin,
  HepMC::GenVertex::particles_in_const_iterator in_end,
  HepMC::GenVertex::particles_out_const_iterator out_begin,
  HepMC::GenVertex::particles_out_const_iterator out_end
) const
{
  ParticleData const * pid;

  std::ostringstream vDescription;
 
  std::size_t j = 0;
 
  HepMC::GenVertex::particles_in_const_iterator in, itmp;
          
  for(in = in_begin; in != in_end; in++, j++)
  {
    if (!j) vDescription << "(";

    HepPDT::ParticleID particleType((*in)->pdg_id());

    if (particleType.isValid())
    {
       pid = pdt_->particle(particleType);
       if (pid)
         vDescription << pid->name();
       else
         vDescription << (*in)->pdg_id();
    }
    else
      vDescription << (*in)->pdg_id();

    itmp = in;

    if (++itmp == in_end) vDescription << ")";
    else vDescription << ",";
  }
        
  vDescription << "->"; j = 0;
 
  HepMC::GenVertex::particles_out_const_iterator out, otmp;
          
  for(out = out_begin; out != out_end; out++, j++)
  {
    if (!j) vDescription << "(";

    HepPDT::ParticleID particleType((*out)->pdg_id());

    if (particleType.isValid())
    {
       pid = pdt_->particle(particleType);
       if (pid)
         vDescription << pid->name();
       else
         vDescription << (*out)->pdg_id();
    }
    else
      vDescription << (*out)->pdg_id();

    otmp = out;

    if (++otmp == out_end) vDescription << ")";
    else vDescription << ",";
  }
        
  return vDescription.str();
}


DEFINE_ANOTHER_FWK_MODULE(SimpleTHA);
