/*
 *  TruthTOA.C
 *
 *  Created by Victor Eduardo Bazterra on 5/31/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "TFile.h"
#include "TH1D.h"

// system include files
#include <iostream>
#include <map>
#include <memory>
#include <string>
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

class TruthTOA : public edm::EDAnalyzer 
{

public:

  explicit TruthTOA(const edm::ParameterSet&);
  ~TruthTOA();

private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // Member data
  
  int matchedHits_;
  
  typedef std::set<int> sint;
  typedef std::vector<double> vdouble;
  typedef std::vector<std::string> vstring;
  typedef std::vector<vstring> vvstring;  
  typedef std::vector<edm::ParameterSet> vParameterSet;
  
  std::string trackingParticleModule_;
  std::string trackingParticleInstance_;
  
  std::string rootFile_;
  bool status_, antiparticles_;

  TrackOrigin tracer_;
    
  vvstring vetoList_;
  
  // Track origin

  struct counter_info_t
  {
    int pdgId;
    int tracks;
     						          
    counter_info_t(int pdgId_, int tracks_)
    {
      pdgId = pdgId_;
      tracks = tracks_;
    }

    bool operator< (const counter_info_t & i) const
    {
      if (pdgId < i.pdgId)
        return true;
      else if (pdgId > i.pdgId)
        return false;
 
      if (tracks < i.tracks)
        return true;
	
      return false;
    }          
  }; 

  typedef std::map<int,int>  counter_index_t;
  typedef std::pair<int,int> counter_index_pair_t;

  typedef std::map<int,std::size_t>  counter_buffer_t;
  typedef std::pair<int,std::size_t> counter_buffer_pair_t;

  typedef std::map<counter_info_t,std::size_t>  counter_t;
  typedef std::pair<counter_info_t,std::size_t> counter_pair_t;
 
  counter_t counter_;
  counter_index_t counter_index_; 
  counter_buffer_t counter_buffer_;
   
  edm::ESHandle<ParticleDataTable> pdt_;
  
  void InitCounter();
  void UpdateCounter();
  void Count(int barcode=0,int pdgId=0);
  
};


TruthTOA::TruthTOA(const edm::ParameterSet& iConfig) : tracer_(iConfig)
{
  trackingParticleModule_ = iConfig.getParameter<std::string> ( "trackingParticleModule" );

  trackingParticleInstance_ = iConfig.getParameter<std::string> ( "trackingParticleProduct" );

  matchedHits_ = iConfig.getParameter<int> ( "matchedHits" );

  rootFile_ = iConfig.getParameter<std::string> ( "rootFile" );
  
  antiparticles_ = iConfig.getParameter<bool> ( "antiparticles" );

  status_ = iConfig.getParameter<bool> ( "status2" );

  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet> ( "veto" );
  
  vstring vetoListNames = pset.getParameterNames();
  
  vetoList_.reserve(vetoListNames.size());
  
  for(std::size_t i=0; i < vetoListNames.size(); i++)
    vetoList_.push_back(pset.getParameter<vstring> (vetoListNames[i]));
}


TruthTOA::~TruthTOA() { }


void
TruthTOA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Tracking particle information
  edm::Handle<TrackingParticleCollection>  TPCollection;
  iEvent.getByLabel(trackingParticleModule_, trackingParticleInstance_, TPCollection);

  // Set the history depth
  if(status_)
    tracer_.depth(-2);
  else
    tracer_.depth(-1);

  // Initialize and reset the temporal counters 
  InitCounter();
  
  // Loop over the track collection.
  for (std::size_t index = 0; index < TPCollection->size(); index++)
  {
    // Get a pointer to a track per each track in collection
    TrackingParticleRef track(TPCollection, index);

    // If the track is not fake then get the orginal particles
    if ((*track).matchedHit() >= matchedHits_)
    {
      if ( tracer_.evaluate(track) )
      {
        const HepMC::GenParticle * particle = tracer_.particle();
        // If the origin can be determined then take the first particle as the original
        if (particle)
          Count(particle->barcode(), particle->pdg_id());
      }
      else
        Count(0,0);
    }
  }
  UpdateCounter();
}   


void 
TruthTOA::beginJob(const edm::EventSetup& iSetup) 
{  
  // Get the particles table.
  iSetup.getData( pdt_ );
}


void 
TruthTOA::endJob() 
{
  TFile file(rootFile_.c_str(), "RECREATE"); 
  file.cd();

  double vetoedVals;
  double totalVals;

  std::cout << "List of all long lived particle found" << std::endl;
  for ( counter_t::iterator it = counter_.begin(); it != counter_.end(); it++)
  {
    if ( !it->first.pdgId )
      std::cout << " fake tracks -> " << it->second << std::endl;    
    else
      std::cout << " particle " <<pdt_->particle(HepPDT::ParticleID(it->first.pdgId))->name(); 
      std::cout << " associated to " << it->first.tracks << " tracks -> " << it->second << std::endl;
  }

  std::multimap<double, std::string> particleTypes;

  for(std::size_t cid = 0; cid < vetoList_.size(); cid++)
  {  
    particleTypes.clear();
    vetoedVals = 0; totalVals = 0;

    for (counter_t::iterator it = counter_.begin(); it != counter_.end(); it++)
    {
      std::ostringstream particle;
       
      if ( !it->first.pdgId )
        particle << "Fake" << std::endl;
      else
        particle << pdt_->particle(HepPDT::ParticleID(it->first.pdgId))->name(); 

      if (
        std::find (
		      vetoList_[cid].begin(),
		      vetoList_[cid].end(),
		      particle.str()
	      ) == vetoList_[cid].end()
      )
      {
        particle << "#" << it->first.tracks;
        particleTypes.insert(std::pair<double,std::string>(it->second, particle.str()));
        totalVals += it->second;
      }
      else 
      {
        vetoedVals += it->second;
        totalVals += it->second;
      }	      
    }

    std::cout << "Veto list #" << cid << std::endl;
    std::cout << " number of vetoed tracks " << vetoedVals << std::endl;
    std::cout << " total number of tracks " << totalVals << std::endl;
    std::cout << " % of vetoed tracks " << ((vetoedVals/totalVals) * 100) << std::endl;
        
    std::ostringstream hName, hTitle;

    hTitle << "Track origins for the whole track collection";
    hName  << "TrackingParticleCollection";
    
    for(std::size_t v=0; v < vetoList_[cid].size(); v++)
    {
      hTitle << "_" << vetoList_[cid][v];
      hName << "_" << vetoList_[cid][v];
    }
	
  	// creating and saving the pie
    TH1D histogram(
      hName.str().c_str(),
      hTitle.str().c_str(),
      particleTypes.size(), 
      0., 
      Double_t(particleTypes.size()) 
    );
	
	  // creating and saving the histogram
    int i = 1;    
    std::cout << "Particle size " <<  particleTypes.size() << std::endl;
    
    std::map<double, std::string>::const_iterator it;

    for(it = particleTypes.begin(); it != particleTypes.end(); it++, i++)
    {
      histogram.GetXaxis()->SetBinLabel(i, it->second.c_str());
      histogram.SetBinContent(i, Double_t(it->first));
    }
    
    histogram.Write();
    file.Flush();
  }
}

void TruthTOA::InitCounter()
{
  counter_buffer_.clear();
  counter_index_.clear();
}

void TruthTOA::Count(int barcode, int pdgId)
{
  counter_buffer_t::iterator it = counter_buffer_.find(barcode);
  
  if ( it != counter_buffer_.end() )
  {
    it->second++;
  }
  else 
  { 
    counter_buffer_.insert( counter_buffer_pair_t(barcode, 1) );
    counter_index_.insert( counter_index_pair_t(barcode, pdgId) );
  }
}


void TruthTOA::UpdateCounter()
{
  counter_buffer_t::const_iterator csi = counter_buffer_.begin();

  std::size_t particleType;

  for (; csi != counter_buffer_.end(); csi++)
  {
    if (antiparticles_)    
      particleType = counter_index_[csi->first];
    else 	
      particleType = abs(counter_index_[csi->first]);
    if ( !particleType )
    {
      counter_info_t info (particleType, 1);
      counter_t::iterator ci = counter_.find( info );
      if ( ci != counter_.end() )
        ci->second += csi->second;
      else 	
        counter_.insert( counter_pair_t (info, csi->second) );
    }
    else
    {
      counter_info_t info = counter_info_t (particleType, csi->second);
      counter_t::iterator ci = counter_.find( info );
      if ( ci != counter_.end() )
        ci->second++;
      else 	
        counter_.insert( counter_pair_t (info, 1) );
    }
  }
}

//define this as a plug-in
DEFINE_ANOTHER_FWK_MODULE(TruthTOA);

