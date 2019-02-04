// File: DataMixingPileupCopy.cc
// Description:  see DataMixingPileupCopy.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------
#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
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

    GenPUProtonsInputTags_ = ps.getParameter<std::vector<edm::InputTag> >("GenPUProtonsInputTags");

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

    // Gen. PU protons
    std::shared_ptr<edm::Wrapper<std::vector<reco::GenParticle> > const> GenPUProtonsPTR;
    for(std::vector<edm::InputTag>::const_iterator it_InputTag = GenPUProtonsInputTags_.begin(); 
                                                   it_InputTag != GenPUProtonsInputTags_.end(); ++it_InputTag){ 
      GenPUProtonsPTR = getProductByTag<std::vector<reco::GenParticle> >( *ep, *it_InputTag , mcc);
      if( GenPUProtonsPTR != nullptr ){
         GenPUProtons_.push_back( *( GenPUProtonsPTR->product() ) );
         GenPUProtons_labels_.push_back( it_InputTag->label() );
      } else edm::LogWarning("DataMixingPileupCopy") << "Missing product with label: " << ( *it_InputTag ).label();
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
    std::unique_ptr<std::vector<PileupSummaryInfo> > PSIVector(new std::vector<PileupSummaryInfo>);
    std::unique_ptr<int> bsInt(new int);

    std::vector<PileupSummaryInfo>::const_iterator PSiter;
    for(PSiter = PileupSummaryStorage_.begin(); PSiter != PileupSummaryStorage_.end(); PSiter++){
      PSIVector->push_back(*PSiter);
    }

    *bsInt=bsStorage_;

    if(FoundPlayback_ ) {
      std::unique_ptr<CrossingFramePlaybackInfoNew> CFPlaybackInfo(new CrossingFramePlaybackInfoNew(CrossingFramePlaybackStorage_));
      e.put(std::move(CFPlaybackInfo));
    }
    e.put(std::move(PSIVector));
    e.put(std::move(bsInt),"bunchSpacing");

    // Gen. PU protons
    for(size_t idx = 0; idx < GenPUProtons_.size(); ++idx){
       std::unique_ptr<std::vector<reco::GenParticle> > GenPUProtons_ptr( new std::vector<reco::GenParticle>() );
       std::vector<reco::GenParticle>::const_iterator it_GenParticle = GenPUProtons_.at(idx).begin();
       std::vector<reco::GenParticle>::const_iterator it_GenParticle_end = GenPUProtons_.at(idx).end();
       for(; it_GenParticle != it_GenParticle_end; ++ it_GenParticle) GenPUProtons_ptr->push_back( *it_GenParticle );

       e.put( std::move(GenPUProtons_ptr), GenPUProtons_labels_.at(idx) );
    }

    // clear local storage after this event
    PileupSummaryStorage_.clear();
    GenPUProtons_.clear();
    GenPUProtons_labels_.clear();
  }
} //edm
