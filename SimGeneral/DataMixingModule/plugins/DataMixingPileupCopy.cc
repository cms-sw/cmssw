// File: DataMixingPileupCopy.cc
// Description:  see DataMixingPileupCopy.h
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
#include "DataMixingPileupCopy.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingPileupCopy::DataMixingPileupCopy() { } 

  // Constructor 
  DataMixingPileupCopy::DataMixingPileupCopy(const edm::ParameterSet& ps) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // Pileup/Playback information

    PileupInfoInputTag_ = ps.getParameter<edm::InputTag>("PileupInfoInputTag");
    CFPlaybackInputTag_ = ps.getParameter<edm::InputTag>("CFPlaybackInputTag");

  }
	       
  // Virtual destructor needed.
  DataMixingPileupCopy::~DataMixingPileupCopy() { 
  }  


  void DataMixingPileupCopy::addPileupInfo(const EventPrincipal *ep, unsigned int eventNr,
                                           ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingPileupCopy") <<"\n===============> adding pileup Info from event  "<<ep->id();

    // find PileupSummaryInfo, CFPlayback information, if it's there

    // Pileup info first

    boost::shared_ptr<Wrapper< std::vector<PileupSummaryInfo> >  const> PileupInfoPTR =
      getProductByTag<std::vector<PileupSummaryInfo>>(*ep,PileupInfoInputTag_, mcc);

    if(PileupInfoPTR ) {

      PileupSummaryStorage_ = *(PileupInfoPTR->product()) ;

      LogDebug("DataMixingEMWorker") << "PileupInfo Size: " << PileupSummaryStorage_.size();

    }

    // Playback

    boost::shared_ptr<Wrapper<CrossingFramePlaybackInfoExtended>  const> PlaybackPTR =
      getProductByTag<CrossingFramePlaybackInfoExtended>(*ep,CFPlaybackInputTag_, mcc);

    FoundPlayback_ = false;

    if(PlaybackPTR ) {

      CrossingFramePlaybackStorage_ = *(PlaybackPTR->product()) ;

      FoundPlayback_ = true;

    }

  }
 
  void DataMixingPileupCopy::putPileupInfo(edm::Event &e) {

    std::auto_ptr<std::vector<PileupSummaryInfo> > PSIVector(new std::vector<PileupSummaryInfo>);

    std::vector<PileupSummaryInfo>::const_iterator PSiter;

    for(PSiter = PileupSummaryStorage_.begin(); PSiter != PileupSummaryStorage_.end(); PSiter++){

      PSIVector->push_back(*PSiter);

    }

    if(FoundPlayback_ ) {

      std::vector<std::vector<edm::EventID> > IdVect; 

      CrossingFramePlaybackStorage_.getEventStartInfo(IdVect, 0);

      std::auto_ptr< CrossingFramePlaybackInfoExtended  > CFPlaybackInfo( new CrossingFramePlaybackInfoExtended(0, IdVect.size(), 1 ));

      CFPlaybackInfo->setEventStartInfo(IdVect, 0);

      e.put(CFPlaybackInfo);

    }

    e.put(PSIVector);


    // clear local storage after this event
    PileupSummaryStorage_.clear();

  }

} //edm
