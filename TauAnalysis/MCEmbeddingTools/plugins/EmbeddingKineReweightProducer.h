#ifndef TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightProducer_h
#define TauAnalysis_MCEmbeddingTools_EmbeddingKineReweightProducer_h

/** \class EmbeddingKineReweightProducer
 *
 * Compute reweighting factors compensating (small) biases of the embedding procedure
 *
 * \authors Christian Veelken
 *
 * \version $Revision: 1.2 $
 *
 * $Id: EmbeddingKineReweightProducer.h,v 1.2 2013/03/06 16:38:00 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

#include <TFile.h>
#include <TH1.h>
#include <TAxis.h>

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
	numDimensions_(0),
	lut_(0),
	xAxis_(0),
	numBinsX_(0),
	yAxis_(0),
	numBinsY_(0)
    {
      if ( variable == "genDiTauPt" ) { 
	variable_ = kGenDiTauPt; 
	numDimensions_ = 1; 
      } else if ( variable == "genDiTauMass" ) { 
	variable_ = kGenDiTauMass;
	numDimensions_ = 1; 
      } else if ( variable == "genDiTauMassVsGenDiTauPt" ) { 
	variable_ = kGenDiTauMass_vs_genDiTauPt;
	numDimensions_ = 2;
      } else if ( variable == "genTau2PtVsGenTau1Pt" ) { 
	variable_ = kGenTau2Pt_vs_genTau1Pt;
	numDimensions_ = 2;
      } else if ( variable == "genTau2EtaVsGenTau1Eta" ) { 
	variable_ = kGenTau2Eta_vs_genTau1Eta;
	numDimensions_ = 2;
      } else throw cms::Exception("EmbeddingKineReweightProducer") 
	  << " Invalid Configuration Parameter 'variable' = " << variable << " !!\n";
      assert(numDimensions_ >= 1 && numDimensions_ <= 2);
      TH1* lut = dynamic_cast<TH1*>(inputFile.Get(lutName.data()));
      if ( !lut ) 
	throw cms::Exception("EmbeddingReweightProducer") 
	  << " Failed to load LUT = " << lutName << " from file = " << inputFile.GetName() << " !!\n";
      lut_ = (TH1*)lut->Clone(std::string(lut->GetName()).append("_cloned").data());
      if ( !lut_->GetSumw2N() ) lut_->Sumw2();
      xAxis_ = lut_->GetXaxis();
      numBinsX_ = xAxis_->GetNbins(); 
      if ( numDimensions_ >= 2 ) {
	yAxis_ = lut_->GetYaxis();
	numBinsY_ = yAxis_->GetNbins();
      }
    }
    ~lutEntryType()
    {
      delete lut_;
    }
    double operator()(const reco::Candidate& genDiTau) const
    {
      double x = 0.;
      double y = 0.;
      if ( variable_ == kGenDiTauPt   ) {
	x = genDiTau.pt();
      } else if ( variable_ == kGenDiTauMass ) {
	x = genDiTau.mass();
      } else if ( variable_ == kGenDiTauMass_vs_genDiTauPt ) {
	x = genDiTau.pt();
	y = genDiTau.mass();
      } else if ( variable_ == kGenTau2Pt_vs_genTau1Pt || variable_ == kGenTau2Eta_vs_genTau1Eta ) {
	bool genTauFound = false;
	const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&genDiTau);
	if ( genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2 ) {
	  const reco::Candidate* genTau1 = genDiTau_composite->daughter(0);
	  const reco::Candidate* genTau2 = genDiTau_composite->daughter(1);
	  if ( genTau1 && genTau2 ) {
	    if ( variable_ == kGenTau2Pt_vs_genTau1Pt ) {
	      x = genTau1->pt();
	      y = genTau2->pt();
	    } else if ( variable_ == kGenTau2Eta_vs_genTau1Eta ) {
	      x = genTau1->eta();
	      y = genTau2->eta();
	    } else assert(0);
	    genTauFound = true;
	  }
	}
	if ( !genTauFound ) {
	  edm::LogWarning ("<EmbeddingKineReweightProducer>")
	    << "Failed to find gen. Tau decay products --> returning weight = 1.0 !!" << std::endl;
	  return 1.;
	}
      } else assert(0);
      int idxBinX = xAxis_->FindBin(x);
      if ( idxBinX <= 1         ) idxBinX = 1;
      if ( idxBinX >= numBinsX_ ) idxBinX = numBinsX_;
      if ( numDimensions_ == 1  ) return lut_->GetBinContent(idxBinX);
      int idxBinY = yAxis_->FindBin(y);
      if ( idxBinY <= 1         ) idxBinY = 1;
      if ( idxBinY >= numBinsY_ ) idxBinY = numBinsY_;
      if ( numDimensions_ == 2  ) return lut_->GetBinContent(idxBinX, idxBinY);
      assert(0);
    }
    enum { kGenDiTauPt, kGenDiTauMass, kGenDiTauMass_vs_genDiTauPt, kGenTau2Pt_vs_genTau1Pt, kGenTau2Eta_vs_genTau1Eta };
    std::string variableName_;
    int variable_;
    int numDimensions_;
    TH1* lut_;
    TAxis* xAxis_;
    int numBinsX_;
    TAxis* yAxis_;
    int numBinsY_;
  };

  std::vector<lutEntryType*> lutEntries_;

  double minWeight_;
  double maxWeight_;

  int verbosity_;
};

#endif

