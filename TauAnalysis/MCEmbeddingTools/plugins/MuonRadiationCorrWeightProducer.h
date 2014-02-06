#ifndef TauAnalysis_MCEmbeddingTools_MuonRadiationCorrWeightProducer_h
#define TauAnalysis_MCEmbeddingTools_MuonRadiationCorrWeightProducer_h

/** \class MuonRadiationCorrWeightProducer
 *
 * Compute reweighting factor and uncertainty
 * related to correction applied to energy and momentum of embedded taus
 * in order to compensate for muon -> muon + photon radiation of reconstructed muons
  *
 * \authors Christian Veelken
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonRadiationCorrWeightProducer.h,v 1.1 2013/01/31 16:15:37 veelken Exp $
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
#include <TH2.h>
#include <TString.h>

#include <string>
#include <map>

class MuonRadiationCorrWeightProducer : public edm::EDProducer 
{
 public:
  explicit MuonRadiationCorrWeightProducer(const edm::ParameterSet&);
  ~MuonRadiationCorrWeightProducer();

  void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag srcMuonsBeforeRad_;
  edm::InputTag srcMuonsAfterRad_;

  struct lutEntryType
  {
    lutEntryType(TFile& inputFile, const std::string& directory, double minMuPlusEn, double maxMuPlusEn, double minMuMinusEn, double maxMuMinusEn)
      : minMuPlusEn_(minMuPlusEn),
	maxMuPlusEn_(maxMuPlusEn),
	minMuMinusEn_(minMuMinusEn),
	maxMuMinusEn_(maxMuMinusEn),
	lut_(0)
    {
      TString lutName = directory;
      if ( lutName.Length() > 0 && !lutName.EndsWith("/") ) lutName.Append("/");
      lutName.Append(directory); // CV: repeat directory, as it is done by GenMuonRadCorrAnalyzer
      if ( lutName.Length() > 0 && !lutName.EndsWith("/") ) lutName.Append("/");
      lutName.Append("genMuonRadCorr");
      if      ( minMuPlusEn_  < 0. && maxMuPlusEn_  < 0. ) lutName.Append("");
      else if (                       maxMuPlusEn_  < 0. ) lutName.Append(Form("_muPlusEnGt%1.0f", minMuPlusEn_));
      else if ( minMuPlusEn_  < 0.                       ) lutName.Append(Form("_muPlusEnLt%1.0f", maxMuPlusEn_));
      else                                                 lutName.Append(Form("_muPlusEn%1.0fto%1.0f", minMuPlusEn_, maxMuPlusEn_));
      if      ( minMuMinusEn_ < 0. && maxMuMinusEn_ < 0. ) lutName.Append("");
      else if (                       maxMuMinusEn_ < 0. ) lutName.Append(Form("_muMinusEnGt%1.0f", minMuMinusEn_));
      else if ( minMuMinusEn_ < 0.                       ) lutName.Append(Form("_muMinusEnLt%1.0f", maxMuMinusEn_));
      else                                                 lutName.Append(Form("_muMinusEn%1.0fto%1.0f", minMuMinusEn_, maxMuMinusEn_));
      TH2* lut = dynamic_cast<TH2*>(inputFile.Get(lutName.Data()));
      if ( !lut ) 
	throw cms::Exception("MuonRadiationCorrWeightProducer") 
	  << " Failed to load LUT = " << lutName << " from file = " << inputFile.GetName() << " !!\n";
      lut_ = (TH2*)lut->Clone(std::string(lut->GetName()).append("_cloned").data());
      if ( !lut_->GetSumw2N() ) lut_->Sumw2();
      if ( lut_->Integral() != 0. ) lut_->Scale(1./lut_->Integral());
    }
    ~lutEntryType() 
    {
      delete lut_;
    }

    bool isWithinBounds(double muPlusEn, double muMinusEn)
    {
      if ( (minMuPlusEn_  <= 0. || muPlusEn  > minMuPlusEn_  ) &&
	   (maxMuPlusEn_  <= 0. || muPlusEn  < maxMuPlusEn_  ) &&
	   (minMuMinusEn_ <= 0. || muMinusEn > minMuMinusEn_ ) &&
	   (maxMuMinusEn_ <= 0. || muMinusEn < maxMuMinusEn_ ) ) return true;
      else return false;
    }
    double getP(const reco::Candidate::LorentzVector& genMuonPlusP4_beforeRad, const reco::Candidate::LorentzVector& genMuonPlusP4_afterRad, 
		const reco::Candidate::LorentzVector& genMuonMinusP4_beforeRad, const reco::Candidate::LorentzVector& genMuonMinusP4_afterRad)
    {
      TAxis* xAxis = lut_->GetXaxis();
      double x = (genMuonPlusP4_beforeRad.E() - genMuonPlusP4_afterRad.E())/genMuonPlusP4_beforeRad.E();    
      int binX = xAxis->FindBin(x);
      if ( binX < 1 ) binX = 1;
      int numBinsX = xAxis->GetNbins();
      if ( binX > numBinsX ) binX = numBinsX;
      
      TAxis* yAxis = lut_->GetYaxis();
      double y = (genMuonMinusP4_beforeRad.E() - genMuonMinusP4_afterRad.E())/genMuonMinusP4_beforeRad.E();
      int binY = yAxis->FindBin(y);
      if ( binY < 1 ) binY = 1;
      int numBinsY = yAxis->GetNbins();
      if ( binY > numBinsY ) binY = numBinsY;
      
      return lut_->GetBinContent(binX, binY);
    }

    double minMuPlusEn_;
    double maxMuPlusEn_;
    double minMuMinusEn_;
    double maxMuMinusEn_;
    
    TH2* lut_;
  };

  typedef std::vector<lutEntryType*> vlutEntryType;

  std::string lutDirectoryRef_;
  vlutEntryType lutEntriesRef_;

  std::map<std::string, std::string>  lutDirectoriesOthers_; // key = name of model 
  std::map<std::string, vlutEntryType> lutEntriesOthers_; // distribution of dEmu1/Emu1 vs. dEmu2/Emu2 for alternative FSR models (e.g. PYTHIA); key = name of model 
  int numOthers_;

  double minWeight_;
  double maxWeight_;

  int verbosity_;

  int numWarnings_;
  int maxWarnings_;
};

#endif

