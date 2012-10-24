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
 * $Id: embeddingAuxFunctions.h,v 1.1 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonReco/interface/Muon.h>

std::vector<const reco::Muon*> getSelMuons(edm::Event&, const edm::InputTag&);

const reco::Muon* getTheMuPlus(const std::vector<const reco::Muon*>&);
const reco::Muon* getTheMuMinus(const std::vector<const reco::Muon*>&);

#endif
