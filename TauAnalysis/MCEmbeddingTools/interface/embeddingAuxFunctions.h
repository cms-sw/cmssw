#ifndef TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h
#define TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h

/**
 *
 * Define methods to retrieve "the" (= highest Pt)
 * muons of positive and negative charge from the event.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: embeddingAuxFunctions.h,v 1.1 2012/10/24 09:37:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>

std::vector<reco::CandidateBaseRef > getSelMuons(edm::Event&, const edm::InputTag&);

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef >&);
reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef >&);

#endif
