#ifndef TauAnalysis_MCEmbeddingTools_MuonRadiationFilter_h
#define TauAnalysis_MCEmbeddingTools_MuonRadiationFilter_h

/** \class MuonRadiationFilter
 *
 * Veto events in which a muon from Z --> mu+ mu- decay radiates a photon:
 *  muon -> muon + photon
 * 
 * \author Christian Veelken, LLR
 *        (based on python code developed by Mike Bachtis)
 *
 * \version $Revision: 1.3 $
 *
 * $Id: MuonRadiationFilter.h,v 1.3 2012/11/25 15:43:13 veelken Exp $
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

class MuonRadiationFilter : public edm::EDFilter 
{
 public:
  explicit MuonRadiationFilter(const edm::ParameterSet&);
  ~MuonRadiationFilter() {}

 private:
  bool filter(edm::Event&, const edm::EventSetup&);

  typedef edm::View<edm::FwdPtr<reco::PFCandidate> > PFView;

  double compCaloEnECAL(const reco::Candidate::LorentzVector&, const PFView&);
  void compPFIso_raw(const reco::Candidate::LorentzVector&, const PFView&, 
		     const reco::Candidate::LorentzVector&, const reco::Candidate::LorentzVector&, 
		     double&, double&, double&);
  double compPFIso_puCorr(const reco::Candidate::LorentzVector&, const PFView&, const PFView&, 
			  const reco::Candidate::LorentzVector&, const reco::Candidate::LorentzVector&);
  bool checkMuonRadiation(const reco::Candidate::LorentzVector&, const reco::Candidate::LorentzVector*, double, const PFView&, const PFView&, 
			  const reco::Candidate&, const reco::Candidate&);

  edm::InputTag srcSelectedMuons_;
  edm::InputTag srcPFCandsNoPU_;
  edm::InputTag srcPFCandsPU_;

  double minPtLow_;
  double dRlowPt_;
  bool addCaloEnECALlowPt_;
  bool applyMassWindowSelectionLowPt_;

  double minPtHigh_;
  double dRhighPt_;
  bool addCaloEnECALhighPt_;
  bool applyMassWindowSelectionHighPt_;

  double dRvetoCone_;
  double dRisoCone_;
  double maxRelIso_;

  double maxMass_;

  bool invert_;
  bool filter_;

  int verbosity_;
};

#endif
