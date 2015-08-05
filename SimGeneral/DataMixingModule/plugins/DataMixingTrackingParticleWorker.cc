// File: DataMixingTrackingParticleWorker.cc
// Description:  see DataMixingTrackingParticleWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
//
//
#include "DataMixingTrackingParticleWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingTrackingParticleWorker::DataMixingTrackingParticleWorker() { }

  // Constructor 
  DataMixingTrackingParticleWorker::DataMixingTrackingParticleWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC)
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    TrackingParticleLabelSig_  = ps.getParameter<edm::InputTag>("TrackingParticleLabelSig");

    TrackingParticlePileInputTag_ = ps.getParameter<edm::InputTag>("TrackingParticlePileInputTag");

    TrackingParticleCollectionDM_  = ps.getParameter<std::string>("TrackingParticleCollectionDM");

    TrackSigToken_ = iC.consumes<std::vector<TrackingParticle> >(TrackingParticleLabelSig_);
    TrackPileToken_ = iC.consumes<std::vector<TrackingParticle> >(TrackingParticlePileInputTag_);

    VtxSigToken_ = iC.consumes<std::vector<TrackingVertex> >(TrackingParticleLabelSig_);
    VtxPileToken_ = iC.consumes<std::vector<TrackingVertex> >(TrackingParticlePileInputTag_);

  }
	       

  // Virtual destructor needed.
  DataMixingTrackingParticleWorker::~DataMixingTrackingParticleWorker() { 
  }  

  // Need an event initialization

  void DataMixingTrackingParticleWorker::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {	

    // Create new track/vertex lists, getting references, too, so that we can cross-link everything

    NewTrackList_ = std::auto_ptr<std::vector<TrackingParticle>>(new std::vector<TrackingParticle>());
    //NewVertexList_ = std::auto_ptr<std::vector<TrackingVertex>>(new std::vector<TrackingVertex>());
    TempVertexList_ = std::vector<TrackingVertex>();

    TrackListRef_  =const_cast<edm::Event&>( e ).getRefBeforePut< std::vector<TrackingParticle> >(TrackingParticleCollectionDM_); 
    VertexListRef_ =const_cast<edm::Event&>( e ).getRefBeforePut< std::vector<TrackingVertex> >(TrackingParticleCollectionDM_);    

  }					   


  void DataMixingTrackingParticleWorker::addTrackingParticleSignals(const edm::Event &e) { 

    // grab Vertices, store copy, preserving indices.  Easier to loop over vertices first - fewer links

    edm::Handle<std::vector<TrackingVertex>> vtxs;
    e.getByToken(VtxSigToken_, vtxs);

    int StartingIndexV = int(TempVertexList_.size());  // should be zero here, but keep for consistency
    int StartingIndexT = int(NewTrackList_->size());  // should be zero here, but keep for consistency

    if (vtxs.isValid()) {
      for (std::vector<TrackingVertex>::const_iterator vtx = vtxs->begin();  vtx != vtxs->end();  ++vtx) {
	TempVertexList_.push_back(*vtx);
      }
    }

    // grab tracks, store copy

    edm::Handle<std::vector<TrackingParticle>> tracks;
    e.getByToken(TrackSigToken_, tracks);

    if (tracks.isValid()) {
      for (std::vector<TrackingParticle>::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	auto oldRef=track->parentVertex();
	auto newRef=TrackingVertexRef( VertexListRef_, oldRef.index()+StartingIndexV );
	NewTrackList_->push_back(*track);

	auto & Ntrack = NewTrackList_->back();  //modify copy

        Ntrack.setParentVertex( newRef );
        Ntrack.clearDecayVertices();

	// next, loop over daughter vertices, same strategy

	for( auto const& vertexRef : track->decayVertices() ) {
	  auto newRef=TrackingVertexRef( VertexListRef_, vertexRef.index()+StartingIndexV );
	  Ntrack.addDecayVertex(newRef);
	}
      }      
    }

    // Now that tracks are handled, go back and put correct Refs in vertices

    for (auto & vertex : TempVertexList_ ) {

      vertex.clearParentTracks();
      vertex.clearDaughterTracks();

      for( auto const& trackRef : vertex.sourceTracks() ) {
        auto newRef=TrackingParticleRef( TrackListRef_, trackRef.index()+StartingIndexT );
        vertex.addParentTrack(newRef);
      }

      // next, loop over daughter tracks, same strategy                                                                 
      for( auto const& trackRef : vertex.daughterTracks() ) {
        auto newRef=TrackingParticleRef( TrackListRef_, trackRef.index()+StartingIndexT );
        vertex.addDaughterTrack(newRef);
      }
    }
  } // end of addTrackingParticleSignals



  void DataMixingTrackingParticleWorker::addTrackingParticlePileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
								    ModuleCallingContext const* mcc) {

    LogDebug("DataMixingTrackingParticleWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    int StartingIndexV = int(TempVertexList_.size());  // keep track of offsets
    int StartingIndexT = int(NewTrackList_->size());  // keep track of offsets

    std::shared_ptr<Wrapper<std::vector<TrackingVertex> >  const> inputVPTR =
      getProductByTag<std::vector<TrackingVertex> >(*ep, TrackingParticlePileInputTag_, mcc);

    if(inputVPTR ) {

      const std::vector<TrackingVertex>  *vtxs = const_cast< std::vector<TrackingVertex> * >(inputVPTR->product());

      // grab vertices, store copy

      for (std::vector<TrackingVertex>::const_iterator vtx = vtxs->begin();  vtx != vtxs->end();  ++vtx) {
	TempVertexList_.push_back(*vtx);
      }
    }


    std::shared_ptr<Wrapper<std::vector<TrackingParticle> >  const> inputPTR =
      getProductByTag<std::vector<TrackingParticle> >(*ep, TrackingParticlePileInputTag_, mcc);

    if(inputPTR ) {

      const std::vector<TrackingParticle>  *tracks = const_cast< std::vector<TrackingParticle> * >(inputPTR->product());

      // grab tracks, store copy
      for (std::vector<TrackingParticle>::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	auto oldRef=track->parentVertex();
	auto newRef=TrackingVertexRef( VertexListRef_, oldRef.index()+StartingIndexV );
	NewTrackList_->push_back(*track);

	auto & Ntrack = NewTrackList_->back();  //modify copy

        Ntrack.setParentVertex( newRef );
        Ntrack.clearDecayVertices();

	// next, loop over daughter vertices, same strategy

	for( auto const& vertexRef : track->decayVertices() ) {
	  auto newRef=TrackingVertexRef( VertexListRef_, vertexRef.index()+StartingIndexV );
	  Ntrack.addDecayVertex(newRef);
	}
      }      
    }

    // Now that tracks are handled, go back and put correct Refs in vertices

    for (auto & vertex : TempVertexList_ ) {

      vertex.clearParentTracks();
      vertex.clearDaughterTracks();

      for( auto const& trackRef : vertex.sourceTracks() ) {
        auto newRef=TrackingParticleRef( TrackListRef_, trackRef.index()+StartingIndexT );
        vertex.addParentTrack(newRef);
      }

      // next, loop over daughter tracks, same strategy                                                                 
      for( auto const& trackRef : vertex.daughterTracks() ) {
        auto newRef=TrackingParticleRef( TrackListRef_, trackRef.index()+StartingIndexT );
        vertex.addDaughterTrack(newRef);
      }
    }


  } // end of addPileups


 
  void DataMixingTrackingParticleWorker::putTrackingParticle(edm::Event &e) {

    // collection of Vertices to put in the event

    NewVertexList_ = std::auto_ptr<std::vector<TrackingVertex>>(new std::vector<TrackingVertex>(TempVertexList_));    

    // put the collection of digis in the event   
    LogInfo("DataMixingTrackingParticleWorker") << "total # Merged Tracks: " << NewTrackList_->size() ;

    // put collections

    e.put( NewTrackList_, TrackingParticleCollectionDM_ );
    e.put( NewVertexList_, TrackingParticleCollectionDM_ );

    // clear local storage for this event
    //NewTrackList_.clear();
    TempVertexList_.clear();
  }

} //edm
