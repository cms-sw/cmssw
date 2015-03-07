// File: DataMixingPileupCopy.cc
// Description:  see DataMixingPileupCopy.h
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
#include "DataMixingPileupCopy.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingPileupCopy::DataMixingPileupCopy() { } 

  // Constructor 
  DataMixingPileupCopy::DataMixingPileupCopy(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // Pileup/Playback information

    PileupInfoInputTag_ = ps.getParameter<edm::InputTag>("PileupInfoInputTag");
    BunchSpacingInputTag_ = ps.getParameter<edm::InputTag>("BunchSpacingInputTag");
    CFPlaybackInputTag_ = ps.getParameter<edm::InputTag>("CFPlaybackInputTag");


    // apparently, we don't need consumes from Secondary input stream
    //iC.consumes<std::vector<PileupSummaryInfo>>(PileupInfoInputTag_);
    //iC.consumes<int>(BunchSpacingInputTag_);
    //iC.consumes<CrossingFramePlaybackInfoNew>(CFPlaybackInputTag_);
  }
	       
  // Virtual destructor needed.
  DataMixingPileupCopy::~DataMixingPileupCopy() { 
  }  


  void DataMixingPileupCopy::addPileupInfo(const EventPrincipal *ep, unsigned int eventNr,
                                           ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingPileupCopy") <<"\n===============> adding pileup Info from event  "<<ep->id();

    // find PileupSummaryInfo, CFPlayback information, if it's there

    // Pileup info first

    std::shared_ptr<Wrapper< std::vector<PileupSummaryInfo> >  const> PileupInfoPTR =
      getProductByTag<std::vector<PileupSummaryInfo>>(*ep,PileupInfoInputTag_, mcc);

    std::shared_ptr<Wrapper< int >  const> bsPTR =
      getProductByTag<int>(*ep,BunchSpacingInputTag_, mcc);

    if(PileupInfoPTR ) {
      PileupSummaryStorage_ = *(PileupInfoPTR->product()) ;
      LogDebug("DataMixingEMWorker") << "PileupInfo Size: " << PileupSummaryStorage_.size();
    }

    if(bsPTR ) {
      bsStorage_ = *(bsPTR->product()) ;
    }
    else {
      bsStorage_=10000;
    }

    // Playback
    std::shared_ptr<Wrapper<CrossingFramePlaybackInfoNew>  const> PlaybackPTR =
      getProductByTag<CrossingFramePlaybackInfoNew>(*ep,CFPlaybackInputTag_, mcc);
    FoundPlayback_ = false;
    if(PlaybackPTR ) {
      CrossingFramePlaybackStorage_ = *(PlaybackPTR->product()) ;
      FoundPlayback_ = true;
    }
  }
 
  void DataMixingPileupCopy::putPileupInfo(edm::Event &e) {
    std::auto_ptr<std::vector<PileupSummaryInfo> > PSIVector(new std::vector<PileupSummaryInfo>);
    std::auto_ptr<int> bsInt(new int);

    std::vector<PileupSummaryInfo>::const_iterator PSiter;
    for(PSiter = PileupSummaryStorage_.begin(); PSiter != PileupSummaryStorage_.end(); PSiter++){
      PSIVector->push_back(*PSiter);
    }

    *bsInt=bsStorage_;

    if(FoundPlayback_ ) {
      std::auto_ptr<CrossingFramePlaybackInfoNew> CFPlaybackInfo(new CrossingFramePlaybackInfoNew(CrossingFramePlaybackStorage_));
      e.put(CFPlaybackInfo);
    }
    e.put(PSIVector);
    e.put(bsInt,"bunchSpacing");

    // clear local storage after this event
    PileupSummaryStorage_.clear();
  }
} //edm
