
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
// $Id: OptTOA.h,v 1.2 2007/05/21 18:06:15 bazterra Exp $
//
//

#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "TFile.h"
#include "TH1F.h"

#include "HepPDT/ParticleID.hh"

// user include files
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoBTag/Analysis/interface/Tools.h"
#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"

#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/TrackHistory/interface/TrackOrigin.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

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
  std::string jetTracksAssociator_;
  
  std::string rootFile_;

  double minPt_, maxChi2_;
  std::size_t minNumberOfHits_;

  bool associationByHits_;
  TrackAssociatorBase * associator_;

  void 
  LoopOverJetTracksAssociation(
    reco::RecoToSimCollection &,
    const edm::ESHandle<MagneticField> &,
    const edm::ESHandle<TransientTrackBuilder> &,
    const edm::Handle<reco::VertexCollection> &,
    const edm::Handle<reco::JetTracksAssociationCollection> &
  );
    
  // Histograms for optimization

  struct histogram_element_t
  {
    double decayLength;
    double distanceToAxis;
    double TIP; // Transverse impact parameter
    double LIP; // Longitudinal impact parameter
    double pt;
    double chi2;
    std::size_t hits;
   
    histogram_element_t(double d, double a, double t, double l, double p, double c, std::size_t h)
    {
      decayLength = d;
      distanceToAxis = a;
      TIP = t;
      LIP = l;
      pt = p;
      chi2 = c;
      hits = h;
    } 
	
    histogram_element_t(const histogram_element_t & orig)
    {
      decayLength = orig.decayLength;
      distanceToAxis = orig.distanceToAxis;
      TIP = orig.TIP;
      LIP = orig.LIP;
      pt = orig.pt;
      chi2 = orig.chi2;
      hits = orig.hits;      
    }	
  };

  typedef std::vector<std::vector<histogram_element_t> > histogram_data_t;
  histogram_data_t histogram_data_;
  
  class histogram_t
  {
  
    TH1F* decayLength;
    TH1F* distanceToAxis;
    TH1F* TIP; // Transverse impact parameter
    TH1F* LIP; // Longitudinal impact parameter
    TH1F* pt;
    TH1F* pt_1gev;
    TH1F* chi2;
    TH1F* hits;
      
  public:
    
    histogram_t(const std::string & particleType)
    {
      std::string name, title;
      name = std::string("hits_") + particleType;
      title = std::string("Hit distribution for ") + particleType;
      hits = new TH1F(name.c_str(), title.c_str(), 13, 4.5, 17.5);
      
      name = std::string("chi2_") + particleType;
      title = std::string("Chi2 distribution for ") + particleType;
      chi2 = new TH1F(name.c_str(), title.c_str(), 100, 0., 30.);

      name = std::string("pt_") + particleType;
      title = std::string("Pt distribution for ") + particleType;
      pt = new TH1F(name.c_str(), title.c_str(), 400, 0., 30.);

      name = std::string("pt_1Gev_") + particleType;
      title = std::string("Pt distribution close 1Gev for ") + particleType;
      pt_1gev = new TH1F(name.c_str(), title.c_str(), 100, 0., 2.);
      
      name = std::string("tip_") + particleType;
      title = std::string("Transverse impact parameter distribution for ") + particleType;
      TIP = new TH1F(name.c_str(), title.c_str(), 100, -0.3, 0.3);

      name = std::string("lip_") + particleType;
      title = std::string("Longitudinal impact parameter distribution for ") + particleType;
      LIP = new TH1F(name.c_str(), title.c_str(), 100, -1., 1.);

      name = std::string("decayLength_") + particleType;
      title = std::string("Decay length distribution for ") + particleType;
      decayLength = new TH1F(name.c_str(), title.c_str(), 100, -5., 5.);

      name = std::string("distanceToAxis_") + particleType;
      title = std::string("Distance to jet distribution for ") + particleType;
      distanceToAxis = new TH1F(name.c_str(), title.c_str(), 100, 0.0, 0.2);
    }
	
    ~histogram_t()
    {
      delete hits;
      delete chi2;
      delete pt;
      delete pt_1gev;    
      delete TIP;
      delete LIP;
      delete decayLength;
      delete distanceToAxis;
    }
    
    void Fill(const histogram_element_t & data)
    {
      hits->Fill(data.hits);
      chi2->Fill(data.chi2);    
      pt->Fill(data.pt);
      pt_1gev->Fill(data.pt);      
      TIP->Fill(data.TIP);
      LIP->Fill(data.LIP);
      decayLength->Fill(data.decayLength);
      distanceToAxis->Fill(data.distanceToAxis);
    }
                    
    void Write()
    {
      hits->Write();
      chi2->Write();    
      pt->Write();
      pt_1gev->Write();      
      TIP->Write();
      LIP->Write();
      decayLength->Write();
      distanceToAxis->Write();
    }
  };
  
  std::string primaryVertex_;

  double d0Pull(
    const edm::ESHandle<MagneticField> &,
    const reco::TrackRef,
    reco::RecoToSimCollection &,
    bool
  );

};


//
// constructors and destructor
//
OptTOA::OptTOA(const edm::ParameterSet& iConfig)
{
  trackCollection_     = iConfig.getParameter<std::string> ( "trackCollection" );
  jetTracksAssociator_ = iConfig.getParameter<std::string> ( "jetTracksAssociator" ); 

  rootFile_ = iConfig.getParameter<std::string> ( "rootFile" );
  
  associationByHits_ = iConfig.getParameter<bool> ( "associationByHits" );

  minPt_ = iConfig.getParameter<double> ( "minPt" );
  maxChi2_ = iConfig.getParameter<double> ( "maxChi2" );
  minNumberOfHits_ = iConfig.getParameter<int> ( "minNumberOfHits" );

  edm::ParameterSet pset;
  
  primaryVertex_ = iConfig.getParameter<std::string> ( "primaryVertex" );
}

OptTOA::~OptTOA() { }

//
// member functions
//

// ------------ method called to for each event  ------------
void
OptTOA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Tracking particle information
  edm::Handle<TrackingParticleCollection>  TPCollection;
  // Track collection
  edm::Handle<reco::TrackCollection> trackCollection;
  // Primary vertex
  edm::Handle<reco::VertexCollection> primaryVertex;
  // Jet to tracks associator
  edm::Handle<reco::JetTracksAssociationCollection> jetTracksAssociator;
  // Trasient track builder
  edm::ESHandle<TransientTrackBuilder> TTbuilder;  
  // Trasient track builder
  edm::ESHandle<MagneticField> theMF;
  
  iEvent.getByType(TPCollection);

  iEvent.getByLabel(trackCollection_,trackCollection);

	iEvent.getByLabel(primaryVertex_, primaryVertex);
	iEvent.getByLabel(jetTracksAssociator_, jetTracksAssociator);
	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", TTbuilder);

  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
   
  reco::RecoToSimCollection association;
 
  if ( !trackCollection_.empty() )
    association = associator_->associateRecoToSim ( trackCollection, TPCollection, &iEvent ); 

  std::cout << std::endl;
  std::cout << "New event" << std::endl;

  LoopOverJetTracksAssociation(
    association,
    theMF,
    TTbuilder,
    primaryVertex,
    jetTracksAssociator
  );
}    


// ------------ method called once each job just before starting event loop  ------------
void 
OptTOA::beginJob(const edm::EventSetup& iSetup) 
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
  reco::RecoToSimCollection & association,
  const edm::ESHandle<MagneticField> & theMF,
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
  	std::vector<reco::Vertex> sortedList =
	  pvs.sortedList(*(primaryVertex.product()));
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

  TrackOrigin tracer(-2);
 
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
      double pt = tracks[index]->pt();
      double chi2 = tracks[index]->normalizedChi2();
      std::size_t hits = tracks[index]->recHitsSize();
        
      if(hits < minNumberOfHits_ || chi2 > maxChi2_ || pt < minPt_ ) continue;
      
      const reco::TransientTrack transientTrack = bproduct->build(&(*tracks[index]));
      double distanceToAxis = - SignedImpactParameter3D::distanceWithJetAxis(transientTrack, direction, pv).second.value();
      double decayLength = SignedDecayLength3D::apply(transientTrack, direction, pv).second.value();
      double d0pull = d0Pull(theMF, tracks[index], association, associationByHits_);
      double dz = tracks[index]->dz() - pvZ;
      double d0 = tracks[index]->d0();
      
      // If the track is not fake then get the orginal particles
      if (tracer.evaluate(tracks[index], association, associationByHits_))
      {
        const HepMC::GenParticle * particle = tracer.particle();
 
        /*
 
        if (particle) 
        { 
          if (fabs(d0pull) > 3.0)  // Badly reconstructed
            histogram_data_[3].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));          
          else if (tracer.isDisplaced()) // Displaced tracks
            histogram_data_[2].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
          else if (HepPDT::ParticleID(particle->pdg_id()).hasBottom()) // B tracks first
            histogram_data_[0].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));                  
          else // Anything else
            histogram_data_[1].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
        }
        else // Particle without image in the generator
          histogram_data_[1].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
  
        */
  
        if (particle) 
        { 
          if (HepPDT::ParticleID(particle->pdg_id()).hasBottom()) // B tracks first
            histogram_data_[0].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));      
          else if (fabs(d0pull) > 3.0)  // Badly reconstructed
            histogram_data_[3].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));          
          else if (tracer.isDisplaced()) // Displaced tracks
            histogram_data_[2].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
          else // Anything else
            histogram_data_[1].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
        }
        else // Particle without image in the generator
          histogram_data_[1].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));

      }
      else
      {
        std::cout << "Fake +++ " << std::endl;
        histogram_data_[4].push_back(histogram_element_t(decayLength, distanceToAxis, d0, dz, pt, chi2, hits));
      }
    }
  }
}


double OptTOA::d0Pull(
  const edm::ESHandle<MagneticField> & theMF,
  const reco::TrackRef track,
  reco::RecoToSimCollection & association,  
  bool associationByHits
)
{
  std::vector<std::pair<TrackingParticleRef, double> > tp;
  
  try
  {
    tp = association[track];
  }
  catch (edm::Exception event)
  {
    return 0;
  }

  // get the track with maximum(minimum) match for associator by hit(chi2)
  double match = 0;
  TrackingParticleRef tpr;
  
  for (std::size_t i=0; i<tp.size(); i++) 
  {
    if (associationByHits) 
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

  //compute tracking particle parameters at point of closest approach to the beamline

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
    theMF.product()
  );

  /*  
  GlobalTrajectoryParameters  theGlobalParameters(
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
    theMF.product()
  );
  
  FreeTrajectoryState ftsAtProduction(theGlobalParameters);
  */
      
  TSCPBuilderNoMaterial tscpBuilder;
  
  TrajectoryStateClosestToPoint tsAtClosestApproach = tscpBuilder(
    ftsAtProduction,
    GlobalPoint(0,0,0)
  ); //as in TrackProducerAlgorithm
  
  GlobalPoint v = tsAtClosestApproach.theState().position();
  GlobalVector p = tsAtClosestApproach.theState().momentum(); 
  
  double d0Sim = - (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));

  return (track->d0()-d0Sim)/track->d0Error();
}  

DEFINE_ANOTHER_FWK_MODULE(OptTOA);
