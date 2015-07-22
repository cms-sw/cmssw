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



  void DataMixingTrackingParticleWorker::addTrackingParticleSignals(const edm::Event &e) { 

    // Create new track/vertex lists; Rely on the fact that addSignals gets called first...

    NewTrackList_ = std::auto_ptr<std::vector<TrackingParticle>>(new std::vector<TrackingParticle>());
    NewVertexList_ = std::auto_ptr<std::vector<TrackingVertex>>(new std::vector<TrackingVertex>());

    // grab tracks, store copy

    //edm::Handle<std::vector<TrackingParticle>> generalTrkHandle;
    //e.getByLabel("generalTracks", generalTrkHandle);
    edm::Handle<std::vector<TrackingParticle>> tracks;
    e.getByToken(TrackSigToken_, tracks);

    if (tracks.isValid()) {
      for (std::vector<TrackingParticle>::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	NewTrackList_->push_back(*track);
      }
      
    }
    // grab Vertices, store copy

    //edm::Handle<std::vector<TrackingVertex>> generalTrkHandle;
    //e.getByLabel("generalTracks", generalTrkHandle);
    edm::Handle<std::vector<TrackingVertex>> vtxs;
    e.getByToken(VtxSigToken_, vtxs);

    if (vtxs.isValid()) {
      for (std::vector<TrackingVertex>::const_iterator vtx = vtxs->begin();  vtx != vtxs->end();  ++vtx) {
	NewVertexList_->push_back(*vtx);
      }
      
    }

  } // end of addTrackingParticleSignals



  void DataMixingTrackingParticleWorker::addTrackingParticlePileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
								    ModuleCallingContext const* mcc) {

    LogDebug("DataMixingTrackingParticleWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;


    std::shared_ptr<Wrapper<std::vector<TrackingParticle> >  const> inputPTR =
      getProductByTag<std::vector<TrackingParticle> >(*ep, TrackingParticlePileInputTag_, mcc);

    if(inputPTR ) {

      const std::vector<TrackingParticle>  *tracks = const_cast< std::vector<TrackingParticle> * >(inputPTR->product());

    // grab tracks, store copy


      for (std::vector<TrackingParticle>::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	NewTrackList_->push_back(*track);
      }

    }

    std::shared_ptr<Wrapper<std::vector<TrackingVertex> >  const> inputVPTR =
      getProductByTag<std::vector<TrackingVertex> >(*ep, TrackingParticlePileInputTag_, mcc);

    if(inputVPTR ) {

      const std::vector<TrackingVertex>  *vtxs = const_cast< std::vector<TrackingVertex> * >(inputVPTR->product());

    // grab vertices, store copy

      for (std::vector<TrackingVertex>::const_iterator vtx = vtxs->begin();  vtx != vtxs->end();  ++vtx) {
	NewVertexList_->push_back(*vtx);
      }

    }

  }


 
  void DataMixingTrackingParticleWorker::putTrackingParticle(edm::Event &e) {

    // collection of Tracks to put in the event

    // put the collection of digis in the event   
    LogInfo("DataMixingTrackingParticleWorker") << "total # Merged Tracks: " << NewTrackList_->size() ;

    // put collections

    e.put( NewTrackList_, TrackingParticleCollectionDM_ );
    e.put( NewVertexList_, TrackingParticleCollectionDM_ );

    // clear local storage for this event
    //NewTrackList_.clear();
    //NewVertexList_.clear();
  }

} //edm
