
// -*- C++ -*-
//
// Package:    OptTOA
// Class:      OptTOA
// 
/**\class OptTOA OptTOA.cc temp/TrackOriginAnalyzerOptTOA/src/OptTOA.cc

 Description: This analyzer calculates the relative composition from a list of
 particle categories.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Victor Bazterra
//         Created:  Tue Mar 13 14:15:40 CDT 2007
// $Id: OptTOA.cc,v 1.6 2008/03/17 22:52:28 bazterra Exp $
//
//

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "TFile.h"
#include "TH1F.h"

#include "HepPDT/ParticleID.hh"

// user include files
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoBTag/Analysis/interface/Tools.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//
// class decleration
//

class OptTOA : public edm::EDAnalyzer 
{

public:

  explicit OptTOA(const edm::ParameterSet&);
  ~OptTOA();

private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // Member data
  
  typedef std::vector<int> vint;
  typedef std::vector<std::string> vstring;

  std::string trackCollection_;
  std::string jetTracks_;
  
  std::string rootFile_;

  int minimumNumberOfHits_, minimumNumberOfPixelHits_;
  double minimumTransverseMomentum_, maximumChiSquared_;

  bool useAllQualities_;
  reco::TrackBase::TrackQuality trackQuality_;

  void 
  LoopOverJetTracksAssociation(
    const edm::ESHandle<TransientTrackBuilder> &,
    const edm::Handle<reco::VertexCollection> &,
    const edm::Handle<reco::JetTracksAssociationCollection> &
  );
    
  // Histograms for optimization

  struct histogram_element_t
  {
    double sdl;  // Signed decay length
    double dta;  // Distance to jet axis
    double tip;  // Transverse impact parameter
    double lip;  // Longitudinal impact parameter
    double ips;  // Impact parameter significance. 
    double pt;   // Transverse momentum
    double chi2; // Chi^2
    std::size_t hits;      // Number of hits
    std::size_t pixelhits; // Number of hits
       
    histogram_element_t(double d, double a, double t, double l, double i, double p, double c, std::size_t h, std::size_t x)
    {
      sdl = d;
      dta = a;
      tip = t;
      lip = l;
      ips = i;
      pt = p;
      chi2 = c;
      hits = h;
      pixelhits = x;
    } 
	
    histogram_element_t(const histogram_element_t & orig)
    {
      sdl = orig.sdl;
      dta = orig.dta;
      tip = orig.tip;
      lip = orig.lip;
      ips = orig.ips;
      pt = orig.pt;
      chi2 = orig.chi2;
      hits = orig.hits;      
      pixelhits = orig.pixelhits;
    }	
  };

  typedef std::vector<std::vector<histogram_element_t> > histogram_data_t;
  histogram_data_t histogram_data_;
  
  class histogram_t
  {
  
    TH1F* sdl;
    TH1F* dta;
    TH1F* tip;
    TH1F* lip;
    TH1F* ips;
    TH1F* pixelhits;
    TH1F* pt_1gev;
    TH1F* chi2;
    TH1F* hits;
      
  public:
    
    histogram_t(const std::string & particleType)
    {
      std::string name, title;
      name = std::string("hits_") + particleType;
      title = std::string("Hit distribution for ") + particleType;
      hits = new TH1F(name.c_str(), title.c_str(), 19, -0.5, 18.5);
      
      name = std::string("chi2_") + particleType;
      title = std::string("Chi2 distribution for ") + particleType;
      chi2 = new TH1F(name.c_str(), title.c_str(), 100, 0., 30.);

      name = std::string("pixelhits_") + particleType;
      title = std::string("Pt distribution for ") + particleType;
      pixelhits = new TH1F(name.c_str(), title.c_str(), 7, -0.5, 6.5);

      name = std::string("pt_1Gev_") + particleType;
      title = std::string("Pt distribution close 1Gev for ") + particleType;
      pt_1gev = new TH1F(name.c_str(), title.c_str(), 100, 0., 2.);
      
      name = std::string("tip_") + particleType;
      title = std::string("Transverse impact parameter distribution for ") + particleType;
      tip = new TH1F(name.c_str(), title.c_str(), 100, -0.3, 0.3);

      name = std::string("lip_") + particleType;
      title = std::string("Longitudinal impact parameter distribution for ") + particleType;
      lip = new TH1F(name.c_str(), title.c_str(), 100, -1., 1.);

      name = std::string("ips_") + particleType;      title = std::string("IPS distribution for ") + particleType;      ips = new TH1F(name.c_str(), title.c_str(), 100, -25.0, 25.0);

      name = std::string("sdl_") + particleType;
      title = std::string("Decay length distribution for ") + particleType;
      sdl = new TH1F(name.c_str(), title.c_str(), 100, -5., 5.);

      name = std::string("dta_") + particleType;
      title = std::string("Distance to jet distribution for ") + particleType;
      dta = new TH1F(name.c_str(), title.c_str(), 100, 0.0, 0.2);
    }
	
    ~histogram_t()
    {
      delete hits;
      delete chi2;
      delete pixelhits;
      delete pt_1gev;    
      delete tip;
      delete lip;
      delete ips;
      delete sdl;
      delete dta;
    }
    
    void Fill(const histogram_element_t & data)
    {
      hits->Fill(data.hits);
      chi2->Fill(data.chi2);    
      pixelhits->Fill(data.pt);
      pt_1gev->Fill(data.pt);
      ips->Fill(data.ips);      
      tip->Fill(data.tip);
      lip->Fill(data.lip);
      sdl->Fill(data.sdl);
      dta->Fill(data.dta);
    }
                    
    void Write()
    {
      hits->Write();
      chi2->Write();    
      pixelhits->Write();
      pt_1gev->Write();
      ips->Write();
      tip->Write();
      lip->Write();
      sdl->Write();
      dta->Write();
    }
  };
  
  std::string primaryVertex_;

  // Track classification.
  TrackCategories classifier_;

};


//
// constructors and destructor
//
OptTOA::OptTOA(const edm::ParameterSet& iConfig) : classifier_(iConfig)
{
  trackCollection_ = iConfig.getParameter<std::string> ( "recoTrackModule" );
  jetTracks_       = iConfig.getParameter<std::string> ( "jetTracks" ); 

  rootFile_ = iConfig.getParameter<std::string> ( "rootFile" );

  minimumNumberOfHits_       = iConfig.getParameter<int> ( "minimumNumberOfHits" );  
  minimumNumberOfPixelHits_  = iConfig.getParameter<int> ( "minimumNumberOfPixelHits" );
  minimumTransverseMomentum_ = iConfig.getParameter<double> ( "minimumTransverseMomentum" );
  maximumChiSquared_         = iConfig.getParameter<double> ( "maximumChiSquared" );

  std::string trackQualityType = iConfig.getParameter<std::string>("trackQualityClass"); //used
  trackQuality_ =  reco::TrackBase::qualityByName(trackQualityType);
  useAllQualities_ = false;

  std::transform(trackQualityType.begin(), trackQualityType.end(), trackQualityType.begin(), (int(*)(int)) std::tolower);
  if (trackQualityType == "any")
  {
  	std::cout << "Using any" << std::endl;
    useAllQualities_ = true;
  }
  primaryVertex_ = iConfig.getParameter<std::string> ( "primaryVertex" );
}

OptTOA::~OptTOA() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void
OptTOA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Track collection
  edm::Handle<edm::View<reco::Track> > trackCollection;
  iEvent.getByLabel(trackCollection_,trackCollection);
  // Primary vertex
  edm::Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByLabel(primaryVertex_, primaryVertex);  
  // Jet to tracks associator
  edm::Handle<reco::JetTracksAssociationCollection> jetTracks;
  iEvent.getByLabel(jetTracks_, jetTracks);
  // Trasient track builder
  edm::ESHandle<TransientTrackBuilder> TTbuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", TTbuilder);
 
  // Setting up event information for the track categories.
  classifier_.newEvent(iEvent, iSetup);

  LoopOverJetTracksAssociation(
    TTbuilder,
    primaryVertex,
    jetTracks
  );
}    


// ------------ method called once each job just before starting event loop  ------------
void 
OptTOA::beginJob(const edm::EventSetup& iSetup) 
{    
  histogram_data_.resize(5);
}


// ------------ method called once each job just after ending the event loop  ------------
void 
OptTOA::endJob() 
{
  TFile file(rootFile_.c_str(), "RECREATE");
  file.cd();

  // saving the histograms
  for(std::size_t i=0; i<5; i++)
  {	
    std::string particle;
    if (i == 0)
      particle = std::string("B_tracks");
    else if (i == 1)
      particle = std::string("nonB_tracks");
    else if (i == 2)
      particle = std::string("displaced_tracks");
    else if (i == 3)
      particle = std::string("bad_tracks");
    else
      particle = std::string("fake_tracks");

    histogram_t histogram(particle);

    for (std::size_t j=0; j<histogram_data_[i].size(); j++)
      histogram.Fill(histogram_data_[i][j]);

    histogram.Write();
  }

  file.Flush();
}


void
OptTOA::LoopOverJetTracksAssociation(
  const edm::ESHandle<TransientTrackBuilder> & TTbuilder,
  const edm::Handle<reco::VertexCollection> & primaryVertex,
  const edm::Handle<reco::JetTracksAssociationCollection> & jetTracksAssociation
)
{
  const TransientTrackBuilder * bproduct = TTbuilder.product();

  // getting the primary vertex
  // use first pv of the collection
  reco::Vertex pv;
 
  if(primaryVertex->size() != 0)
  {
    PrimaryVertexSorter pvs;
    std::vector<reco::Vertex> sortedList = pvs.sortedList(*(primaryVertex.product()));
    pv = (sortedList.front());
  }
  else 
  { // create a dummy PV
    // cout << "NO PV FOUND" << endl;
    reco::Vertex::Error e;
    e(0,0)=0.0015*0.0015;
    e(1,1)=0.0015*0.0015;
    e(2,2)=15.*15.;
    reco::Vertex::Point p(0,0,0);
    pv = reco::Vertex(p,e,1,1,1);
  }
  
  reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
  
  int i=0;
  
  for(; it != jetTracksAssociation->end(); it++, i++)
  {
    // get jetTracks object
    reco::JetTracksAssociationRef jetTracks(jetTracksAssociation, i); 

    double pvZ = pv.z(); 
    GlobalVector direction(jetTracks->first->px(), jetTracks->first->py(), jetTracks->first->pz());
      
    // get the tracks associated to the jet
    reco::TrackRefVector tracks = jetTracks->second;
    for(std::size_t index = 0; index < tracks.size(); index++)
	{
	  edm::RefToBase<reco::Track> track(tracks[index]);

	  double pt = tracks[index]->pt();
	  double chi2 = tracks[index]->normalizedChi2();	  
	  int hits = tracks[index]->hitPattern().numberOfValidHits();
	  int pixelHits = tracks[index]->hitPattern().numberOfValidPixelHits();
	  	  
	  if( 
	      hits < minimumNumberOfHits_ || 
              pixelHits < minimumNumberOfPixelHits_ ||
              pt < minimumTransverseMomentum_ || 
              chi2 >  maximumChiSquared_ ||
              (!useAllQualities_ && !tracks[index]->quality(trackQuality_))
          ) continue;
	  
	  const reco::TransientTrack transientTrack = bproduct->build(&(*tracks[index]));
	  double dta = - IPTools::jetTrackDistance(transientTrack, direction, pv).second.value();
	  double sdl = IPTools::signedDecayLength3D(transientTrack, direction, pv).second.value();
      double ips = IPTools::signedImpactParameter3D(transientTrack, direction, pv).second.value();
	  double d0 = IPTools::signedTransverseImpactParameter(transientTrack, direction, pv).second.value();
      double dz = tracks[index]->dz() - pvZ;
	  
	  // Classify the reco track;
	  if ( classifier_.evaluate(edm::RefToBase<reco::Track>(tracks[index])) )
	  {
	    if ( classifier_.is(TrackCategories::Fake) )
	      histogram_data_[4].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
        else if ( classifier_.is(TrackCategories::Bottom) )
		  histogram_data_[0].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));      
        else if ( classifier_.is(TrackCategories::Bad) )
          histogram_data_[3].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));          
        else if ( classifier_.is(TrackCategories::Displaced) )
          histogram_data_[2].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
	    else
	      histogram_data_[1].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));	    
      }
	}
  }
}

DEFINE_ANOTHER_FWK_MODULE(OptTOA);
