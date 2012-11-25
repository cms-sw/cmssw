#ifndef TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h
#define TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h

/**
 *
 * Define methods to retrieve "the" (= highest Pt)
 * muons of positive and negative charge from the event.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: embeddingAuxFunctions.h,v 1.2 2012/10/25 14:41:38 aburgmei Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

std::vector<reco::CandidateBaseRef> getSelMuons(edm::Event&, const edm::InputTag&);

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef>&);
reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef>&);

TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event&, const edm::EventSetup&, TrackDetectorAssociator&, const TrackAssociatorParameters&, const reco::Candidate*);

#endif
