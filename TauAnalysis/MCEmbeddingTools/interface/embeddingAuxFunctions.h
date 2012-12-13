#ifndef TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h
#define TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h

/**
 *
 * Define methods to retrieve "the" (= highest Pt)
 * muons of positive and negative charge from the event.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: embeddingAuxFunctions.h,v 1.3 2012/11/25 15:43:12 veelken Exp $
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

bool matchMuonDetId(uint32_t, uint32_t);
void printMuonDetId(const edm::EventSetup&, uint32_t);

#endif
