// -*- C++ -*-
//
// Package:    ZmumuPFEmbedder
// Class:      ZmumuPFEmbedder
// 
/**\class ZmumuPFEmbedder ZmumuPFEmbedder.cc MyAna/ZmumuPFEmbedder/src/ZmumuPFEmbedder.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Dec  9 16:14:56 CET 2009
// $Id: ZmumuPFEmbedder.cc,v 1.1 2010/03/17 16:14:10 fruboes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>
#include <DataFormats/Candidate/interface/CompositeRefCandidate.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include <DataFormats/Math/interface/deltaR.h>
//
// class decleration
//

class ZmumuPFEmbedder : public edm::EDProducer {
   public:
      explicit ZmumuPFEmbedder(const edm::ParameterSet&);
      ~ZmumuPFEmbedder();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      void produceTrackColl(edm::Event&, const std::vector< reco::PFCandidate > & toBeAdded );
      virtual void endJob() ;
      
      double _etaMax;
      double _ptMin;
      edm::InputTag _tracks;
      
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
ZmumuPFEmbedder::ZmumuPFEmbedder(const edm::ParameterSet& iConfig)
  : _etaMax(iConfig.getUntrackedParameter<double>("etaMax")),
    _ptMin(iConfig.getUntrackedParameter<double>("ptMin")),
    _tracks(iConfig.getParameter<edm::InputTag>("tracks"))
{

   //register your products
   produces< std::vector< reco::Muon >  >("zMusExtracted"); // 
   produces< std::vector< reco::PFCandidate >  >("forMixing"); // 
   produces<reco::TrackCollection>();

   
 //  produces< edm::RefToBaseVector< reco::Candidate >  >(); // 
  // Yp    produces< std::vector< reco::RecoCandidate >  >();
  //  produces< std::vector< reco::CompositeRefCandidate >  >();

   //if do put with a label
   //produces<ExampleData2>("label");
   //now do what ever other initialization is needed
  
}


ZmumuPFEmbedder::~ZmumuPFEmbedder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ZmumuPFEmbedder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   std::cout << "#######"  << std::endl;
   Handle<PFCandidateCollection> pfIn;
   iEvent.getByLabel("particleFlow",pfIn);
   // iterate over pf, choose 2 muons

   std::auto_ptr<std::vector< reco::Muon > > pOut(new std::vector< reco::Muon  > );
   std::auto_ptr<std::vector< reco::PFCandidate > > forMix(new std::vector< reco::PFCandidate  > );
   
   
   std::vector< reco::PFCandidate > toBeAdded;
 
   
   PFCandidateConstIterator it = pfIn->begin();
   PFCandidateConstIterator itE = pfIn->end();
   for (;it!=itE;++it) {
     int pdg = std::abs( it->pdgId() );
     
     if ( pdg == 13 && std::abs(it->eta()) < _etaMax &&  it->pt() > _ptMin ) {
       
       if (toBeAdded.size() < 2 ) {
         toBeAdded.push_back(*it);
       } else if (toBeAdded.size() == 2 ) {
          forMix->push_back( *(toBeAdded.rbegin())  );
          toBeAdded.pop_back();
          toBeAdded.push_back(*it);
       } else { // never should get here
         throw cms::Exception("yadayada") << " Internal err \n";
       }
       
       // keep two elements sorted
       if (toBeAdded.size() == 2 && (toBeAdded.at(0).pt() < toBeAdded.at(1).pt()) ) {
          reco::PFCandidate tmp = *( toBeAdded.begin() );
          toBeAdded.at(0) = toBeAdded.at(1);
          toBeAdded.at(1) = tmp;
       }
       
     } else {
       forMix->push_back(*it);
     }
   }


   Handle<reco::VertexCollection> primaryVertices;
   iEvent.getByLabel("offlinePrimaryVertices", primaryVertices);
   const reco::Vertex vtx = *(primaryVertices->begin());
  
   std::vector< reco::PFCandidate >::iterator itTBA = toBeAdded.begin();
   
   reco::Vertex::Point p = vtx.position();
   for (; itTBA != toBeAdded.end(); ++itTBA) {
       reco::Muon mu(itTBA->charge(), itTBA->p4(), p);
       pOut->push_back(mu); //xxx
   }

   produceTrackColl(iEvent, toBeAdded);
   iEvent.put(pOut, "zMusExtracted");
   iEvent.put(forMix, "forMixing");


   
}


// produces clean track collection wo muon tracks.
void ZmumuPFEmbedder::produceTrackColl(edm::Event & iEvent, const std::vector< reco::PFCandidate > & toBeAdded )
{
   edm::Handle<reco::TrackCollection> tks;
   iEvent.getByLabel( _tracks, tks);

   std::auto_ptr< reco::TrackCollection  > newCol(new reco::TrackCollection );

   double epsilon = 0.00001;
   int nMatched = 0;
   
   for ( reco::TrackCollection::const_iterator it = tks->begin() ; it != tks->end(); ++it) 
   {
     bool ok = true;
     for ( std::vector< reco::PFCandidate >::const_iterator itTBA = toBeAdded.begin();
                                                            itTBA != toBeAdded.end();
                                                            ++itTBA)
     {
       const reco::MuonRef& muonRef = itTBA->muonRef();
       //( muonRef.isNonnull() );
       reco::TrackRef track = muonRef->innerTrack();
       double dr = reco::deltaR( *it, *track);
       //std::cout << "TTTT " << dr << std::endl;
       if (dr < epsilon) {
         ++ nMatched;
         ok = false;
       }
     }
     if (ok)  newCol->push_back(*it);  
   }
   if (nMatched!=2) std::cout << "TTT ARGGGHGH " << nMatched << std::endl;

   iEvent.put(newCol);

}
// ------------ method called once each job just before starting event loop  ------------
void 
ZmumuPFEmbedder::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZmumuPFEmbedder::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZmumuPFEmbedder);
