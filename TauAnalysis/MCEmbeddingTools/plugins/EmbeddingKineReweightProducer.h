#ifndef TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightProducer_h
#define TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightProducer_h

/** \class EmbeddingKineReweightProducer
 *
 * Compute reweighting factors compensating (small) biases of the embedding procedure
 *
 * \authors Christian Veelken
 *
 * \version $Revision: 1.1 $
 *
 * $Id: EmbeddingReweightProducer.h,v 1.1 2013/01/31 16:15:37 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <TFile.h>
#include <TH1.h>

#include <string>
#include <map>

class EmbeddingKineReweightProducer : public edm::EDProducer 
{
 public:
  explicit EmbeddingKineReweightProducer(const edm::ParameterSet&);
  ~EmbeddingKineReweightProducer();

  void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag srcGenDiTaus_;

  struct lutEntryType
  {
    lutEntryType(TFile& inputFile, const std::string& variable, const std::string& lutName)
      : variableName_(variable),
	lut_(0)
    {
      if      ( variable == "genDiTauPt"   ) variable_ = genDiTauPt;
      else if ( variable == "genDiTauMass" ) variable_ = genDiTauMass;
      else throw cms::Exception("EmbeddingKineReweightProducer") 
	<< " Invalid Configuration Parameter 'variable' = " << variable << " !!\n";
      TH1* lut = dynamic_cast<TH1*>(inputFile.Get(lutName.data()));
      if ( !lut ) 
	throw cms::Exception("EmbeddingReweightProducer") 
	  << " Failed to load LUT = " << lutName << " from file = " << inputFile.GetName() << " !!\n";
      lut_ = (TH1*)lut->Clone(std::string(lut->GetName()).append("_cloned").data());
      if ( !lut_->GetSumw2N() ) lut_->Sumw2();
      numBins_ = lut_->GetNbinsX(); 
    }
    ~lutEntryType()
    {
      delete lut_;
    }
    double operator()(const reco::Candidate::LorentzVector& genDiTauP4) const
    {
      double x = 0.;
      if      ( variable_ == genDiTauPt   ) x = genDiTauP4.pt();
      else if ( variable_ == genDiTauMass ) x = genDiTauP4.mass();
      else assert(0);
      int idxBin = lut_->FindBin(x);
      if ( idxBin <= 1        ) idxBin = 1;
      if ( idxBin <= numBins_ ) idxBin = numBins_;
      return lut_->GetBinContent(idxBin);
    }
    enum { genDiTauPt, genDiTauMass };
    std::string variableName_;
    int variable_;
    TH1* lut_;
    int numBins_;
  };

  std::vector<lutEntryType*> lutEntries_;

  double minWeight_;
  double maxWeight_;

  int verbosity_;
};

#endif

