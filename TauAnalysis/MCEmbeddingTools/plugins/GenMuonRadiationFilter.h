#ifndef TauAnalysis_MCEmbeddingTools_GenMuonRadiationFilter_h
#define TauAnalysis_MCEmbeddingTools_GenMuonRadiationFilter_h

/** \classGen MuonRadiationFilter
 *
 * Flag events in which a muon from Z --> mu+ mu- decay radiates a photon:
 *  muon -> muon + photon
 * (on generator level)
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: GenMuonRadiationFilter.h,v 1.3 2012/11/25 15:43:13 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/View.h"

class GenMuonRadiationFilter : public edm::EDFilter 
{
 public:
  explicit GenMuonRadiationFilter(const edm::ParameterSet&);
  ~GenMuonRadiationFilter() {}

 private:
  bool filter(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcGenParticles_;

  double minPtLow_;
  double dRlowPt_;

  double minPtHigh_;
  double dRhighPt_;

  bool invert_;
  bool filter_;

  int numWarnings_;
  int maxWarnings_;

  int verbosity_;
};

#endif
