
/** \class MuonTrackCleaner
 *
 * Produce collection of reco::Tracks in Z --> mu+ mu- event
 * from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * \authors Tomasz Maciej Frueboes;
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MuonTrackCleaner.cc,v 1.2 2013/02/10 12:46:23 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>

class MuonTrackCleaner : public edm::EDProducer 
{
 public:
  explicit MuonTrackCleaner(const edm::ParameterSet&);
  ~MuonTrackCleaner() {}

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcSelectedMuons_;
  typedef std::vector<edm::InputTag> VInputTag;
  VInputTag srcTracks_;

  double dRmatch_;
  bool removeDuplicates_;

  enum { kInnerTrack, kOuterTrack, kLink, kTeV };
  int type_; 
  
  typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, reco::TrackCollection> > TrackToTrackMap;
  bool produceTrackToTrackMap_;

  int maxWarnings_tooMany_;
  int numWarnings_tooMany_;
  int maxWarnings_tooFew_;
  int numWarnings_tooFew_;

  int verbosity_;
};

MuonTrackCleaner::MuonTrackCleaner(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    srcTracks_(cfg.getParameter<VInputTag>("tracks")),
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
  
  produceTrackToTrackMap_ = cfg.getParameter<bool>("produceTrackToTrackMap");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  
  for ( VInputTag::const_iterator srcTracks_i = srcTracks_.begin();
	srcTracks_i != srcTracks_.end(); ++srcTracks_i ) {
    produces<reco::TrackCollection>(srcTracks_i->instance());
    if ( produceTrackToTrackMap_ ) produces<TrackToTrackMap>(srcTracks_i->instance());
  }
}

namespace
{
  struct muonToTrackMatchInfoType
  {
    muonToTrackMatchInfoType(const reco::Particle::LorentzVector& muonP4, const reco::Track* track, double dR)
      : muonPt_(muonP4.pt()),
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

void MuonTrackCleaner::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<MuonTrackCleaner::produce>:" << std::endl;

  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::vector<reco::Particle::LorentzVector> selMuonP4s;
  if ( muPlus.isNonnull() ) {
    if ( verbosity_ ) std::cout << " muPlus: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
    selMuonP4s.push_back(muPlus->p4());
  }
  if ( muMinus.isNonnull() ) {
    if ( verbosity_ ) std::cout << " muMinus: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
    selMuonP4s.push_back(muMinus->p4());
  }

//--- produce collection of reco::Tracks excluding muons
  for ( VInputTag::const_iterator srcTracks_i = srcTracks_.begin();
	srcTracks_i != srcTracks_.end(); ++srcTracks_i ) {
    if ( verbosity_ ) std::cout << "processing trackCollection = " << (*srcTracks_i) << std::endl;
    
    edm::Handle<reco::TrackCollection> tracks;
    evt.getByLabel(*srcTracks_i, tracks);
    
    std::auto_ptr<reco::TrackCollection> tracks_woMuons(new reco::TrackCollection());
    
    std::vector<muonToTrackMatchInfoType> selMuonToTrackMatches;
    for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s.begin();
	  selMuonP4 != selMuonP4s.end(); ++selMuonP4 ) {
      std::vector<muonToTrackMatchInfoType> tmpMatches;
      for ( reco::TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track ) {
	double dR = reco::deltaR(track->eta(), track->phi(), selMuonP4->eta(), selMuonP4->phi());
	if ( dR < dRmatch_ ) tmpMatches.push_back(muonToTrackMatchInfoType(*selMuonP4, &(*track), dR));
      }
      // rank muon-to-track matches by quality
      std::sort(tmpMatches.begin(), tmpMatches.end(), SortMuonToTrackMatchInfosDescendingMatchQuality());
      if ( tmpMatches.size() > 0 ) selMuonToTrackMatches.push_back(tmpMatches.front());
      if ( removeDuplicates_ ) {
	// CV: remove all high Pt tracks very close to muon direction
	//    (duplicate tracks arise in case muon track in SiStrip + Pixel detector is reconstructed as 2 disjoint segments)
	for ( std::vector<muonToTrackMatchInfoType>::const_iterator tmpMatch =  tmpMatches.begin();
	      tmpMatch !=  tmpMatches.end(); ++tmpMatch ) {
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
      if ( verbosity_ && track->pt() > 10. ) {
	std::cout << "track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << ", isMuon = " << isMuon << std::endl;
      }
      if ( isMuon ) removedTracks.push_back(track); // track belongs to a selected muon, do not copy
      else tracks_woMuons->push_back(*track);
    }
    if ( (type_ >= kInnerTrack && type_ <= kLink       && removedTracks.size() > selMuons.size() && numWarnings_tooMany_ < maxWarnings_tooMany_) &&
	 (type_ >= kInnerTrack && type_ <= kInnerTrack && removedTracks.size() < selMuons.size() && numWarnings_tooFew_  < maxWarnings_tooFew_ ) ) {
      edm::LogWarning("MuonTrackCleaner") 
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
    
    evt.put(tracks_woMuons, srcTracks_i->instance());

    if ( produceTrackToTrackMap_ ) {
      std::auto_ptr<TrackToTrackMap> trackToTrackMap_woMuons(new TrackToTrackMap());
      
      edm::Handle<TrackToTrackMap> trackToTrackMap;
      evt.getByLabel(*srcTracks_i, trackToTrackMap);
      for ( typename TrackToTrackMap::const_iterator entry = trackToTrackMap->begin();
	    entry != trackToTrackMap->end(); ++entry ) {
	bool isMuon = false;
	for ( std::vector<reco::TrackRef>::const_iterator removedTrack = removedTracks.begin();
	      removedTrack != removedTracks.end(); ++removedTrack ) {
	  if ( entry->key == (*removedTrack) ) isMuon = true;
	}
	if ( !isMuon ) trackToTrackMap_woMuons->insert(*entry);
      }
      
      evt.put(trackToTrackMap_woMuons, srcTracks_i->instance());
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonTrackCleaner);
