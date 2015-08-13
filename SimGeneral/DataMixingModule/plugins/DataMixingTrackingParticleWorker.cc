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

namespace {
  template <typename T>
  void appendDetSetVector(edm::DetSetVector<T>& target, const edm::DetSetVector<T>& source) {
    for(auto& detsetSource: source) {
      auto& detsetTarget = target.find_or_insert(detsetSource.detId());
      std::copy(detsetSource.begin(), detsetSource.end(), std::back_inserter(detsetTarget));
    }
  }
}

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

    // Pixel and Strip DigiSimlinks

    StripLinkPileInputTag_ = ps.getParameter<edm::InputTag>("StripDigiSimLinkPileInputTag");
    PixelLinkPileInputTag_ = ps.getParameter<edm::InputTag>("PixelDigiSimLinkPileInputTag");
    StripLinkCollectionDM_ = ps.getParameter<std::string>("StripDigiSimLinkCollectionDM");
    PixelLinkCollectionDM_ = ps.getParameter<std::string>("PixelDigiSimLinkCollectionDM");

    StripLinkSigToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(ps.getParameter<edm::InputTag>("StripDigiSimLinkLabelSig"));
    StripLinkPileToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(StripLinkPileInputTag_);
    PixelLinkSigToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink> >(ps.getParameter<edm::InputTag>("PixelDigiSimLinkLabelSig"));
    PixelLinkPileToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink> >(PixelLinkPileInputTag_);

    // Muon DigiSimLinks

    DTLinkPileInputTag_ = ps.getParameter<edm::InputTag>("DTDigiSimLinkPileInputTag");
    RPCLinkPileInputTag_ = ps.getParameter<edm::InputTag>("RPCDigiSimLinkPileInputTag");
    CSCWireLinkPileInputTag_ = ps.getParameter<edm::InputTag>("CSCWireDigiSimLinkPileInputTag");
    CSCStripLinkPileInputTag_ = ps.getParameter<edm::InputTag>("CSCStripDigiSimLinkPileInputTag");

    DTLinkCollectionDM_ = ps.getParameter<std::string>("DTDigiSimLinkDM");
    RPCLinkCollectionDM_ = ps.getParameter<std::string>("RPCDigiSimLinkDM");
    CSCWireLinkCollectionDM_ = ps.getParameter<std::string>("CSCWireDigiSimLinkDM");
    CSCStripLinkCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiSimLinkDM");

    CSCWireLinkSigToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(ps.getParameter<edm::InputTag>("CSCWireDigiSimLinkLabelSig"));
    CSCWireLinkPileToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(CSCWireLinkPileInputTag_);
    CSCStripLinkSigToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(ps.getParameter<edm::InputTag>("CSCStripDigiSimLinkLabelSig"));
    CSCStripLinkPileToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(CSCStripLinkPileInputTag_);
    DTLinkSigToken_ = iC.consumes< MuonDigiCollection<DTLayerId, DTDigiSimLink> >(ps.getParameter<edm::InputTag>("DTDigiSimLinkLabelSig"));
    DTLinkPileToken_ = iC.consumes< MuonDigiCollection<DTLayerId, DTDigiSimLink> >(DTLinkPileInputTag_);
    RPCLinkSigToken_ = iC.consumes<edm::DetSetVector<RPCDigiSimLink> >(ps.getParameter<edm::InputTag>("RPCDigiSimLinkLabelSig"));
    RPCLinkPileToken_ = iC.consumes<edm::DetSetVector<RPCDigiSimLink> >(RPCLinkPileInputTag_);

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

    // tracker

    NewStripLinkList_ = std::make_unique<edm::DetSetVector<StripDigiSimLink> >();
    NewPixelLinkList_ = std::make_unique<edm::DetSetVector<PixelDigiSimLink> >();

    // muons

    NewCSCStripLinkList_ = std::make_unique<edm::DetSetVector<StripDigiSimLink> >();
    NewCSCWireLinkList_ = std::make_unique<edm::DetSetVector<StripDigiSimLink> >();
    NewRPCLinkList_ = std::make_unique<edm::DetSetVector<RPCDigiSimLink> >();
    NewDTLinkList_ = std::make_unique< MuonDigiCollection<DTLayerId, DTDigiSimLink> >();

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

    // Accumulate DigiSimLinks
    edm::Handle<edm::DetSetVector<StripDigiSimLink> > stripLinks;
    e.getByToken(StripLinkSigToken_, stripLinks);
    if(stripLinks.isValid()) {
      appendDetSetVector(*NewStripLinkList_, *stripLinks);
    }

    edm::Handle<edm::DetSetVector<PixelDigiSimLink> > pixelLinks;
    e.getByToken(PixelLinkSigToken_, pixelLinks);
    if(pixelLinks.isValid()) {
      appendDetSetVector(*NewPixelLinkList_, *pixelLinks);
    }

    edm::Handle<edm::DetSetVector<StripDigiSimLink> > CSCstripLinks;
    e.getByToken(CSCStripLinkSigToken_, CSCstripLinks);
    if(CSCstripLinks.isValid()) {
      appendDetSetVector(*NewCSCStripLinkList_, *CSCstripLinks);
    }

    edm::Handle<edm::DetSetVector<StripDigiSimLink> > CSCwireLinks;
    e.getByToken(CSCWireLinkSigToken_, CSCwireLinks);
    if(CSCwireLinks.isValid()) {
      appendDetSetVector(*NewCSCWireLinkList_, *CSCwireLinks);
    }

    edm::Handle<edm::DetSetVector<RPCDigiSimLink> > RPCLinks;
    e.getByToken(RPCLinkSigToken_, RPCLinks);
    if(RPCLinks.isValid()) {
      appendDetSetVector(*NewRPCLinkList_, *RPCLinks);
    }

    edm::Handle< DTDigiSimLinkCollection > DTLinks;
    e.getByToken(DTLinkSigToken_, DTLinks);
    if(DTLinks.isValid()) {
      for (DTDigiSimLinkCollection::DigiRangeIterator detUnit=DTLinks->begin(); detUnit !=DTLinks->end(); ++detUnit) {
	const DTLayerId& layerid = (*detUnit).first;
	const DTDigiSimLinkCollection::Range& range = (*detUnit).second;
	NewDTLinkList_->put(range,layerid);
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


    // Accumulate DigiSimLinks
    std::shared_ptr<Wrapper<edm::DetSetVector<StripDigiSimLink> > const> inputStripPtr =
      getProductByTag<edm::DetSetVector<StripDigiSimLink> >(*ep, StripLinkPileInputTag_, mcc);
    if(inputStripPtr) {
      appendDetSetVector(*NewStripLinkList_, *(inputStripPtr->product()));
    }

    std::shared_ptr<Wrapper<edm::DetSetVector<PixelDigiSimLink> > const> inputPixelPtr =
      getProductByTag<edm::DetSetVector<PixelDigiSimLink> >(*ep, PixelLinkPileInputTag_, mcc);
    if(inputPixelPtr) {
      appendDetSetVector(*NewPixelLinkList_, *(inputPixelPtr->product()));
    }

    std::shared_ptr<Wrapper<edm::DetSetVector<StripDigiSimLink> > const> CSCinputStripPtr =
      getProductByTag<edm::DetSetVector<StripDigiSimLink> >(*ep, CSCStripLinkPileInputTag_, mcc);
    if(CSCinputStripPtr) {
      appendDetSetVector(*NewCSCStripLinkList_, *(CSCinputStripPtr->product()));
    }

    std::shared_ptr<Wrapper<edm::DetSetVector<StripDigiSimLink> > const> CSCinputWirePtr =
      getProductByTag<edm::DetSetVector<StripDigiSimLink> >(*ep, CSCWireLinkPileInputTag_, mcc);
    if(CSCinputWirePtr) {
      appendDetSetVector(*NewCSCWireLinkList_, *(CSCinputWirePtr->product()));
    }

    std::shared_ptr<Wrapper<edm::DetSetVector<RPCDigiSimLink> > const> inputRPCPtr =
      getProductByTag<edm::DetSetVector<RPCDigiSimLink> >(*ep, RPCLinkPileInputTag_, mcc);
    if(inputRPCPtr) {
      appendDetSetVector(*NewRPCLinkList_, *(inputRPCPtr->product()));
    }

    std::shared_ptr<Wrapper< DTDigiSimLinkCollection > const> inputDTPtr =
      getProductByTag< DTDigiSimLinkCollection >(*ep, DTLinkPileInputTag_, mcc);
    if(inputDTPtr) {      
      const DTDigiSimLinkCollection*  DTLinks = const_cast< DTDigiSimLinkCollection * >(inputDTPtr->product());
      for (DTDigiSimLinkCollection::DigiRangeIterator detUnit=DTLinks->begin(); detUnit !=DTLinks->end(); ++detUnit) {
	const DTLayerId& layerid = (*detUnit).first;
	const DTDigiSimLinkCollection::Range& range = (*detUnit).second;
	NewDTLinkList_->put(range,layerid);
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

    e.put( std::move(NewStripLinkList_), StripLinkCollectionDM_ );
    e.put( std::move(NewPixelLinkList_), PixelLinkCollectionDM_ );

    e.put( std::move(NewCSCStripLinkList_), CSCStripLinkCollectionDM_ );
    e.put( std::move(NewCSCWireLinkList_), CSCWireLinkCollectionDM_ );
    e.put( std::move(NewRPCLinkList_), RPCLinkCollectionDM_ );
    e.put( std::move(NewDTLinkList_), DTLinkCollectionDM_ );


    // clear local storage for this event
    //NewTrackList_.clear();
    TempVertexList_.clear();
  }

} //edm
