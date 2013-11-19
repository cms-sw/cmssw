#ifndef TauAnalysis_MCEmbeddingTools_GenParticlesFromZsSelectorForMCEmbedding_h
#define TauAnalysis_MCEmbeddingTools_GenParticlesFromZsSelectorForMCEmbedding_h

/** \class GenParticlesFromZsSelectorForMCEmbedding
 *
 * Select tau leptons, muons and electrons
 * produced in Z/gamma* --> l+ l- processes
 *
 * NOTE: Can handle case that virtual Z/gamma is missing in GenEVT record
 * 
 * CV: This class is a copy of the copy of the code in
 *       TauAnalysis/GenSimTools/plugins/GenParticlesFromZsSelector.h
 *       TauAnalysis/GenSimTools/plugins/GenParticlesFromZsSelector.cc
 *     The copy was created to avoid package dependencies
 *    (of either TauAnalysis/MCEmbeddingTools on TauAnalysis/GenSimTools or
 *     of TauAnalysis/GenSimTools on TauAnalysis/MCEmbeddingTools... I know this is not nice)
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: GenParticlesFromZsSelectorForMCEmbedding.h,v 1.2 2013/01/31 09:07:18 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

class GenParticlesFromZsSelectorForMCEmbedding : public edm::EDProducer 
{
 public:

  explicit GenParticlesFromZsSelectorForMCEmbedding(const edm::ParameterSet&);

  ~GenParticlesFromZsSelectorForMCEmbedding();
  
  void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::InputTag src_;

  typedef std::vector<int> vint;
  vint pdgIdsMothers_;
  vint pdgIdsDaughters_;

  int maxDaughters_;
  int minDaughters_;

  enum { kBeforeFSR, kAfterFSR };
  int before_or_afterFSR_;

  int verbosity_;
};

#endif
