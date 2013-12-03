#ifndef TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightNtupleProducer_h  
#define TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightNtupleProducer_h

/** \class EmbeddingKineReweightNtupleProducer
 *
 * Produce Ntuple for computing embeddingKineReweight LUTs
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: EmbeddingKineReweightNtupleProducer.h,v 1.2 2013/04/17 07:11:37 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <TTree.h>

#include <map>
#include <string>
#include <vector>
#include <ostream>

class EmbeddingKineReweightNtupleProducer : public edm::EDAnalyzer
{
 public:
  
  EmbeddingKineReweightNtupleProducer(const edm::ParameterSet&);
  ~EmbeddingKineReweightNtupleProducer();

  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:

  void addBranchF(const std::string&);
  void addBranchI(const std::string&);
  void addBranch_EnPxPyPz(const std::string&);
  void addBranch_MEtResProjections(const std::string&);

  void printBranches(std::ostream&);

  void setValueF(const std::string&, double);
  void setValueI(const std::string&, int);
  void setValue_EnPxPyPz(const std::string&, const reco::Candidate::LorentzVector&);
  void setValue_MEtResProjections(const std::string&, const reco::Candidate::LorentzVector& , const reco::Candidate::LorentzVector& , const reco::Candidate::LorentzVector&);

  std::string moduleLabel_;

  edm::InputTag srcGenDiTaus_;
  
  edm::InputTag srcGenParticles_;
  edm::InputTag srcSelectedMuons_;
 
  edm::InputTag srcRecLeg1_;
  edm::InputTag srcRecLeg2_;

  edm::InputTag srcRecJets_;

  edm::InputTag srcGenCaloMEt_;
  edm::InputTag srcRecCaloMEt_;
  edm::InputTag srcGenPFMEt_;
  edm::InputTag srcRecPFMEt_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcWeights_;
  edm::InputTag srcGenFilterInfo_;

  struct branchEntryType
  {
    Float_t valueF_;
    Int_t valueI_;
  };

  typedef std::map<std::string, branchEntryType> branchMap; // key = branch name
  branchMap branches_;

  TTree* ntuple_;

  static int verbosity_;
};

#endif


