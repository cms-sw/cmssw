#include "TauAnalysis/MCEmbeddingTools/interface/MuonTrackCleanerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <algorithm>

MuonTrackCleanerBase::MuonTrackCleanerBase(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    dRmatch_(cfg.getParameter<double>("dRmatch")),
    removeDuplicates_(cfg.getParameter<bool>("removeDuplicates")),
    maxWarnings_tooMany_(100),
    numWarnings_tooMany_(0),
    maxWarnings_tooFew_(3),
    numWarnings_tooFew_(0)
{
  std::string type_string = cfg.getParameter<std::string>("type");
  // CV: types defined in RecoMuon/MuonIdentification/python/muons1stStep_cfi.py
  if      ( type_string == "inner tracks" ) type_ = kInnerTrack;
  else if ( type_string == "outer tracks" ) type_ = kOuterTrack;
  else if ( type_string == "links"        ) type_ = kLink;
  else if ( type_string.find("tev") == 0  ) type_ = kTeV;
  else throw cms::Exception("Configuration") 
    << "Invalid Configuration Parameter 'type' = " << type_string << " !!\n";
  
  typedef std::vector<edm::InputTag> VInputTag;
  VInputTag srcTracks = cfg.getParameter<VInputTag>("tracks");
  for ( VInputTag::const_iterator srcTracks_i = srcTracks.begin();
	srcTracks_i != srcTracks.end(); ++srcTracks_i ) {
    todoListEntryType todoListEntry;
    todoListEntry.srcTracks_ = (*srcTracks_i);
    
    todoList_.push_back(todoListEntry); 

    produces<reco::TrackCollection>(todoListEntry.srcTracks_.instance());
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

namespace
{
  struct muonToTrackMatchInfoType
  {
    muonToTrackMatchInfoType(double muonPt, const reco::Track* track, double dR)
      : muonPt_(muonPt),
	trackPt_(track->pt()),
	trackCharge_(track->charge()),
	dR_(dR),
	track_(track)
    {}
    ~muonToTrackMatchInfoType() {}
    double muonPt_;
    double trackPt_;    
    int trackCharge_;
    double dR_;
    const reco::Track* track_;
  };

  struct SortMuonToTrackMatchInfosDescendingMatchQuality
  {
    bool operator() (const muonToTrackMatchInfoType& m1, const muonToTrackMatchInfoType& m2)
    {
      // 1st criterion: prefer matches of high Pt
      if ( m1.trackPt_ > (0.5*m1.muonPt_) && m2.trackPt_ < (0.5*m2.muonPt_) ) return true;  // m1 has higher rank than m2
      if ( m1.trackPt_ < (0.5*m1.muonPt_) && m2.trackPt_ > (0.5*m2.muonPt_) ) return false; // m2 has higher rank than m1
      // 2nd criterion: in case multiple matches to high Pt tracks exist,
      //                take track matched most closely in dR
      return (m1.dR_ < m2.dR_); 
    }
  };

  std::string runLumiEventNumbers_to_string(const edm::Event& evt)
  {
    edm::RunNumber_t run_number = evt.id().run();
    edm::LuminosityBlockNumber_t ls_number = evt.luminosityBlock();
    edm::EventNumber_t event_number = evt.id().event();
    std::ostringstream retVal;
    retVal << "Run = " << run_number << ", LS = " << ls_number << ", Event = " << event_number;
    return retVal.str();
  }
}

void MuonTrackCleanerBase::produce(edm::Event& evt, const edm::EventSetup& es)
{
  produceTracks(evt, es);
  produceTrackExtras(evt, es);
}

MuonTrackCleanerBase::muonMomentumType MuonTrackCleanerBase::getMuonMomentum(const reco::Candidate& muon_candidate)
{
  muonMomentumType muonMomentum;
  muonMomentum.pt_  = muon_candidate.pt();
  muonMomentum.eta_ = muon_candidate.eta();
  muonMomentum.phi_ = muon_candidate.phi();

  const reco::Muon* muon = dynamic_cast<const reco::Muon*>(&muon_candidate);
  if ( muon ) {
    if ( type_ == kInnerTrack && muon->innerTrack().isNonnull() ) {
      muonMomentum.eta_ = muon->innerTrack()->eta();
      muonMomentum.phi_ = muon->innerTrack()->phi();
    } else if ( type_ == kOuterTrack && muon->outerTrack().isNonnull() ) {
      muonMomentum.eta_ = muon->outerTrack()->eta();
      muonMomentum.phi_ = muon->outerTrack()->phi();
    } else if ( (type_ == kLink || type_ == kTeV) && muon->globalTrack().isNonnull() ) {
      muonMomentum.eta_ = muon->globalTrack()->eta();
      muonMomentum.phi_ = muon->globalTrack()->phi();
    }
  } 

  return muonMomentum;
}

void MuonTrackCleanerBase::produceTracks(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<MuonTrackCleanerBase::produceTracks>:" << std::endl;

  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::vector<muonMomentumType> selMuonMomenta;
  if ( muPlus.isNonnull() ) {
    if ( verbosity_ ) std::cout << " muPlus: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
    selMuonMomenta.push_back(getMuonMomentum(*muPlus));
  }
  if ( muMinus.isNonnull() ) {
    if ( verbosity_ ) std::cout << " muMinus: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
    selMuonMomenta.push_back(getMuonMomentum(*muMinus));
  }

//--- produce collection of reco::Tracks excluding muons
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    if ( verbosity_ ) std::cout << "processing trackCollection = " << todoItem->srcTracks_ << std::endl;
    
    todoItem->trackRefMap_.clear();
    
    edm::Handle<reco::TrackCollection> tracks;
    evt.getByLabel(todoItem->srcTracks_, tracks);
    
    std::auto_ptr<reco::TrackCollection> tracks_cleaned(new reco::TrackCollection());

    reco::TrackRefProd trackCollectionRefProd_cleaned = evt.getRefBeforePut<reco::TrackCollection>(todoItem->srcTracks_.instance());
    size_t idxTrack_cleaned = 0;
    
    std::vector<muonToTrackMatchInfoType> selMuonToTrackMatches;
    for ( std::vector<muonMomentumType>::const_iterator selMuonMomentum = selMuonMomenta.begin();
	  selMuonMomentum != selMuonMomenta.end(); ++selMuonMomentum ) {
      std::vector<muonToTrackMatchInfoType> tmpMatches;
      for ( reco::TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track ) {
	double dR = reco::deltaR(track->eta(), track->phi(), selMuonMomentum->eta_, selMuonMomentum->phi_);
	if ( dR < dRmatch_ ) tmpMatches.push_back(muonToTrackMatchInfoType(selMuonMomentum->pt_, &(*track), dR));
      }
      // rank muon-to-track matches by quality
      std::sort(tmpMatches.begin(), tmpMatches.end(), SortMuonToTrackMatchInfosDescendingMatchQuality());
      if ( tmpMatches.size() > 0 ) selMuonToTrackMatches.push_back(tmpMatches.front());
      if ( removeDuplicates_ ) {
	// CV: remove all high Pt tracks very close to muon direction
	//    (duplicate tracks arise in case muon track in SiStrip + Pixel detector is reconstructed as 2 disjoint segments)
	for ( std::vector<muonToTrackMatchInfoType>::const_iterator tmpMatch = tmpMatches.begin();
	      tmpMatch != tmpMatches.end(); ++tmpMatch ) {
	  if ( (tmpMatch->dR_ < 1.e-3 && tmpMatch->trackPt_ >           (0.33*tmpMatch->muonPt_))      ||
	       (tmpMatch->dR_ < 1.e-1 && tmpMatch->trackPt_ > TMath::Max(0.66*tmpMatch->muonPt_, 10.)) ) selMuonToTrackMatches.push_back(*tmpMatch);
	}
      }
    }
    
    std::vector<reco::TrackRef> removedTracks;
    size_t numTracks = tracks->size();
    for ( size_t trackIdx = 0; trackIdx < numTracks; ++trackIdx ) {
      reco::TrackRef track(tracks, trackIdx);
      bool isMuon = false;
      for ( std::vector<muonToTrackMatchInfoType>::const_iterator muonMatchInfo = selMuonToTrackMatches.begin();
	    muonMatchInfo != selMuonToTrackMatches.end(); ++muonMatchInfo ) {
	if ( muonMatchInfo->track_ == &(*track) ) isMuon = true;
      }
      if ( verbosity_ && track->pt() > 5. ) {
	std::cout << "track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << ", isMuon = " << isMuon << std::endl;
      }
      if ( isMuon ) {
	removedTracks.push_back(track); // track belongs to a selected muon, do not copy
      } else { 
	tracks_cleaned->push_back(*track);
	todoItem->trackRefMap_[reco::TrackRef(trackCollectionRefProd_cleaned, idxTrack_cleaned)] = track;
	++idxTrack_cleaned;
      }
    }
    if ( (type_ >= kInnerTrack && type_ <= kLink       && removedTracks.size() > selMuons.size() && numWarnings_tooMany_ < maxWarnings_tooMany_) &&
	 (type_ >= kInnerTrack && type_ <= kInnerTrack && removedTracks.size() < selMuons.size() && numWarnings_tooFew_  < maxWarnings_tooFew_ ) ) {
      edm::LogWarning("MuonTrackCleanerBase") 
	<< " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
	<< " Removed " << removedTracks.size() << " tracks from event containing " << selMuons.size() << " muons !!" << std::endl;
      if ( muPlus.isNonnull() ) std::cout << " muPlus: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
      if ( muMinus.isNonnull() ) std::cout << " muMinus: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
      std::cout << "Removed tracks:" << std::endl;
      int idx = 0;
      for ( std::vector<reco::TrackRef>::const_iterator removedTrack = removedTracks.begin();
	    removedTrack != removedTracks.end(); ++removedTrack ) {
	std::cout << "track #" << idx << ":" 
		  << " Pt = " << (*removedTrack)->pt() << ", eta = " << (*removedTrack)->eta() << ", phi = " << (*removedTrack)->phi() << std::endl;
	++idx;
      }
      if ( removedTracks.size() > selMuons.size() ) ++numWarnings_tooMany_;
      if ( removedTracks.size() < selMuons.size() ) ++numWarnings_tooFew_;
    }
    
    evt.put(tracks_cleaned, todoItem->srcTracks_.instance());
  }
}
