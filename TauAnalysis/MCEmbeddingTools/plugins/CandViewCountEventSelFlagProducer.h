#ifndef TauAnalysis_MCEmbeddingTools_CandViewCountEventSelFlagProducer_h
#define TauAnalysis_MCEmbeddingTools_CandViewCountEventSelFlagProducer_h

/** \class CandViewCountEventSelFlagProducer
 *
 * Produce boolean flag indicating whether collection
 * specified by configuration parameter contains
 * at least 'min' and at most 'max' entries.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CandViewCountEventSelFlagProducer.h,v 1.1 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionFilterTrait.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountEventSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"

class CandViewCountEventSelFlagProducer : public edm::EDProducer
{
 public:
  explicit CandViewCountEventSelFlagProducer(const edm::ParameterSet&);
  ~CandViewCountEventSelFlagProducer() {}
  
 private:
  void produce(edm::Event&, const edm::EventSetup&);

  typedef edm::View<reco::Candidate> CandidateView;
  ObjectCountEventSelector<CandidateView, AnySelector, MinNumberSelector> eventSelector_;
};
 
#endif
