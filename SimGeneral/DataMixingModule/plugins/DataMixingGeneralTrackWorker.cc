// File: DataMixingGeneralTrackWorker.cc
// Description:  see DataMixingGeneralTrackWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
//
//
#include "DataMixingGeneralTrackWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingGeneralTrackWorker::DataMixingGeneralTrackWorker() { }

  // Constructor 
  DataMixingGeneralTrackWorker::DataMixingGeneralTrackWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC)
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    GeneralTrackLabelSig_  = ps.getParameter<edm::InputTag>("GeneralTrackLabelSig");

    GeneralTrackPileInputTag_ = ps.getParameter<edm::InputTag>("GeneralTrackPileInputTag");

    GeneralTrackCollectionDM_  = ps.getParameter<std::string>("GeneralTrackDigiCollectionDM");

    GTrackSigToken_ = iC.consumes<reco::TrackCollection>(GeneralTrackLabelSig_);
    GTrackPileToken_ = iC.consumes<reco::TrackCollection>(GeneralTrackPileInputTag_);

  }
	       

  // Virtual destructor needed.
  DataMixingGeneralTrackWorker::~DataMixingGeneralTrackWorker() { 
  }  



  void DataMixingGeneralTrackWorker::addGeneralTrackSignals(const edm::Event &e) { 

    // Create new track list; Rely on the fact that addSignals gets called first...

    NewTrackList_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());

    // grab tracks, store copy

    //edm::Handle<reco::TrackCollection> generalTrkHandle;
    //e.getByLabel("generalTracks", generalTrkHandle);
    edm::Handle<reco::TrackCollection> tracks;
    e.getByToken(GTrackSigToken_, tracks);

    if (tracks.isValid()) {
      for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	NewTrackList_->push_back(*track);
      }
      
    }

  } // end of addGeneralTrackSignals



  void DataMixingGeneralTrackWorker::addGeneralTrackPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                            ModuleCallingContext const* mcc) {
    LogDebug("DataMixingGeneralTrackWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;


    boost::shared_ptr<Wrapper<reco::TrackCollection >  const> inputPTR =
      getProductByTag<reco::TrackCollection >(*ep, GeneralTrackPileInputTag_, mcc);

    if(inputPTR ) {

      const reco::TrackCollection  *tracks = const_cast< reco::TrackCollection * >(inputPTR->product());

    // grab tracks, store copy


      for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
	NewTrackList_->push_back(*track);
      }
      
    }

  }


 
  void DataMixingGeneralTrackWorker::putGeneralTrack(edm::Event &e) {

    // collection of Tracks to put in the event

    // put the collection of digis in the event   
    LogInfo("DataMixingGeneralTrackWorker") << "total # Merged Tracks: " << NewTrackList_->size() ;

    // put collection

    e.put( NewTrackList_, GeneralTrackCollectionDM_ );

    // clear local storage for this event
    //NewTrackList_.clear();
  }

} //edm
