/*
 *  VertexTHA.C
 *
 *  Created by Victor Eduardo Bazterra on 5/31/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */
 
// system include files
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1D.h"

// user include files

#include "SimTracker/TrackHistory/interface/TrackOrigin.h"

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
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
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

//
// class decleration
//

class VertexTHA : public edm::EDAnalyzer 
{

public:

  explicit VertexTHA(const edm::ParameterSet&);
  ~VertexTHA();

private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // Member data
  
  typedef std::set<int> sint;
  typedef std::vector<double> vdouble;
  typedef std::vector<std::string> vstring;
  typedef std::vector<vstring> vvstring;  
  typedef std::vector<edm::ParameterSet> vParameterSet;

  std::string trackCollection_;
  
  std::string rootFile_;
    
  vvstring vetoList_;

  bool associationByHits_;
  TrackAssociatorBase * associator_;
  
  // Track history

  typedef std::map<std::string,std::size_t>  counter_t;
  typedef std::pair<std::string,std::size_t> counter_pair_t;
  
  counter_t counter_;

  double sourceCut_;
  std::size_t totalTracks_;
   
  edm::ESHandle<ParticleDataTable> pdt_;
  
  void Count(const std::string &);  
};

VertexTHA::VertexTHA(const edm::ParameterSet& iConfig)
{
  trackCollection_ = iConfig.getParameter<std::string> ( "trackCollection" );

  rootFile_ = iConfig.getParameter<std::string> ( "rootFile" );
  
  associationByHits_ = iConfig.getParameter<bool> ( "associationByHits" );
  sourceCut_ = iConfig.getParameter<double> ( "sourceCut" );
}

VertexTHA::~VertexTHA() { }

void
VertexTHA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Tracking particle information
  edm::Handle<TrackingParticleCollection>  TPCollection;
  // Track collection
  edm::Handle<reco::TrackCollection> trackCollection;

  // Get tracking particle information from the file.
  iEvent.getByType(TPCollection);

  // Get reco::TrackCollection from the file.
  iEvent.getByLabel(trackCollection_,trackCollection);

  // Get the associator between reco::Track and TrakingParticle
  reco::RecoToSimCollection association 
    = associator_->associateRecoToSim ( trackCollection, TPCollection, &iEvent ); 

  std::cout << std::endl;
  std::cout << "New event" << std::endl;

  // Initialive the TrackHistory object.    
  TrackOrigin tracer;
  
  // Loop over the track collection.
  for (std::size_t index = 0; index < trackCollection->size(); index++)
  {
    // Get a pointer to a track per each track in collection
    reco::TrackRef track(trackCollection, index);

    // If the track is not fake then get the vertexes
    if (tracer.evaluate(track, association, associationByHits_))
    {
      // Get the list of TrackingVertexes associated to
      TrackingVertexContainer vertexes(tracer.simVertexTrail());
         
      // Loop over all vertexes                       
      if( !vertexes.empty() )
      {
        // create a description of the vertex 
        std::ostringstream vDescription;

        TrackingParticleRefVector tracks(vertexes[0]->sourceTracks());
        std::size_t nTracks = tracks.size();
        ParticleData const * pid;
                
        for(std::size_t j = 0; j < nTracks; j++)
        {
          if (!j) vDescription << "(";

          HepPDT::ParticleID particleType(tracks[j]->pdgId());

          if (particleType.isValid())
          {
            pid = pdt_->particle(particleType);
            if (pid)
              vDescription << pid->name();
            else
              vDescription << '*';            
          }
          else
            vDescription << '*';

          if (j == nTracks - 1) vDescription << ")";
          else vDescription << ",";
        }
        
        vDescription << "->";
        
        tracks = vertexes[0]->daughterTracks();
        nTracks = tracks.size();

        for(std::size_t j = 0; j < nTracks; j++)
        {
          if (!j) vDescription << "(";
          
          HepPDT::ParticleID particleType(tracks[j]->pdgId());

          if (particleType.isValid())
          {
            pid = pdt_->particle(particleType);
            if (pid)
              vDescription << pid->name();
            else
              vDescription << '*';            
          }
          else
            vDescription << '*';

          if (j == nTracks - 1) vDescription << ")";
          else vDescription << ",";
        }
        
        vDescription << '#' << vertexes.size();
        std::cout << "Found associated vertex : " << vDescription.str() << std::endl;
       
        // count the the vertex.
        Count(vDescription.str()); 
      }
      else
        Count(std::string("WithoutTrackingVertexes"));
    }
    else
      Count(std::string("Fake"));
  }
}


void 
VertexTHA::beginJob(const edm::EventSetup& iSetup) 
{
  // Get the associator by hits
  edm::ESHandle<TrackAssociatorBase> associator;

  if(associationByHits_)
  {  
    iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",associator);
    associator_ = (TrackAssociatorBase *) associator.product();
  }
  else  
  {
    iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",associator);
    associator_ = (TrackAssociatorBase *) associator.product();  
  }
  
  // Get the particles table.
  iSetup.getData( pdt_ );
  
  totalTracks_ = 0;
}


void 
VertexTHA::endJob() 
{
  TFile file(rootFile_.c_str(), "RECREATE"); 
  file.cd();

  std::cout << "List of all found vertexes" << std::endl;
  for ( counter_t::iterator it = counter_.begin(); it != counter_.end(); it++)
      std::cout << " Vertex " << it->first << " -> " << it->second << " tracks" << std::endl;

  std::multimap<double, std::string> vDescriptions;

  double cut;

  for (counter_t::iterator it = counter_.begin(); it != counter_.end(); it++)
  {
    cut = (double) it->second/totalTracks_;
    if (cut > sourceCut_)
      vDescriptions.insert(std::pair<double,std::string>(cut, it->first));
  }

  std::ostringstream hName, hTitle;

  hTitle << "TrackingVertexes that originate whole track collection";
  hName  << "vertexTrackHistory";
            
  // creating and saving the pie
  TH1D histogram(
    hName.str().c_str(),
    hTitle.str().c_str(),
    vDescriptions.size(), 
    0., 
    Double_t(vDescriptions.size()) 
  );

  // creating and saving the pie
  TH1D histogram_plus_error(
    (hName.str()+std::string("_plus_error")).c_str(),
    hTitle.str().c_str(),
    vDescriptions.size(), 
    0., 
    Double_t(vDescriptions.size()) 
  );

  // creating and saving the pie
  TH1D histogram_minus_error(
    (hName.str()+std::string("_minus_error")).c_str(),
    hTitle.str().c_str(),
    vDescriptions.size(),
    0.,
    Double_t(vDescriptions.size()) 
  );

  int i = 1;    
  double error;
  
  std::map<double, std::string>::iterator it;
  for(it = vDescriptions.begin(); it != vDescriptions.end(); it++, i++)
  {
    error = sqrt(it->first*(1 - it->first)/totalTracks_);
  
    histogram.GetXaxis()->SetBinLabel(i, it->second.c_str());
    histogram.SetBinContent(i, Double_t(it->first));
    histogram.SetBinError(i, Double_t(error));

    histogram_plus_error.GetXaxis()->SetBinLabel(i, it->second.c_str());
    histogram_plus_error.SetBinContent(i, Double_t(it->first+error));

    histogram_minus_error.GetXaxis()->SetBinLabel(i, it->second.c_str());
    histogram_minus_error.SetBinContent(i, Double_t(it->first-error));
  }
    
  histogram.Write();
  histogram_plus_error.Write();
  histogram_minus_error.Write();  

  file.Flush();
}

void VertexTHA::Count(const std::string & vDescription)
{
  counter_t::iterator it = counter_.find(vDescription);
  
  if ( it != counter_.end() )
    it->second++;
  else 
    counter_.insert( counter_pair_t(vDescription, 1) );

  totalTracks_++;
}

DEFINE_ANOTHER_FWK_MODULE(VertexTHA);


