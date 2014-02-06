#ifndef TauAnalysis_MCEmbeddingTools_MuonTrackCleanerBase_h
#define TauAnalysis_MCEmbeddingTools_MuonTrackCleanerBase_h

/** \class MuonTrackCleanerBase
 *
 * Base class for modules producing collections of reco::Tracks 
 * in Z --> mu+ mu- event from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonTrackCleanerBase.h,v 1.1 2013/03/30 16:41:11 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>
#include <string>

class MuonTrackCleanerBase : public edm::EDProducer 
{
 public:
  explicit MuonTrackCleanerBase(const edm::ParameterSet&);
  ~MuonTrackCleanerBase() {}

 protected:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  struct muonMomentumType
  {
    double pt_;
    double eta_;
    double phi_;
  };
  muonMomentumType getMuonMomentum(const reco::Candidate&);

  virtual void produceTracks(edm::Event&, const edm::EventSetup&);
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&) = 0;

  edm::InputTag srcSelectedMuons_;
  
  struct todoListEntryType
  {
    edm::InputTag srcTracks_;

    mutable std::map<reco::TrackRef, reco::TrackRef> trackRefMap_; // key = edm::Ref to output track collection, value = edm::Ref to input track collections
                                                                   // (needed by GlobalMuonTrackCleaner)
  };
  std::vector<todoListEntryType> todoList_; 

  double dRmatch_;
  bool removeDuplicates_;

  enum { kInnerTrack, kOuterTrack, kLink, kTeV };
  int type_; 

  int maxWarnings_tooMany_;
  int numWarnings_tooMany_;
  int maxWarnings_tooFew_;
  int numWarnings_tooFew_;

  int verbosity_;
};

#endif
