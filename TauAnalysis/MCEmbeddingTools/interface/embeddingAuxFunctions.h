#ifndef TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h
#define TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h

/**
 *
 * Define methods to retrieve "the" (= highest Pt)
 * muons of positive and negative charge from the event.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: embeddingAuxFunctions.h,v 1.5 2012/12/18 15:59:25 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

std::vector<reco::CandidateBaseRef> getSelMuons(const edm::Event&, const edm::InputTag&);

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef>&);
reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef>&);

TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event&, const edm::EventSetup&, TrackDetectorAssociator&, const TrackAssociatorParameters&, const reco::Candidate*);

bool matchMuonDetId(uint32_t, uint32_t);
void printMuonDetId(const edm::EventSetup&, uint32_t);

double getDeDxForPbWO4(double p);

static const double DENSITY_PBWO4 = 8.28;
static const double DENSITY_BRASS = 8.53;
static const double DENSITY_IRON = 7.87;

#endif
