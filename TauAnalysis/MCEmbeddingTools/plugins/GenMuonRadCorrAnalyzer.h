#ifndef TauAnalysis_MCEmbeddingTools_GenMuonRadCorrAnalyzer_h
#define TauAnalysis_MCEmbeddingTools_GenMuonRadCorrAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/GenMuonRadiationAlgorithm.h"

#include <TH2.h>

#include <string>
#include <vector>

class GenMuonRadCorrAnalyzer : public edm::EDAnalyzer 
{
 public:
  explicit GenMuonRadCorrAnalyzer(const edm::ParameterSet&);
  ~GenMuonRadCorrAnalyzer();
    
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:
  edm::InputTag srcSelectedMuons_;
  edm::InputTag srcGenParticles_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcWeights_;

  std::string directory_;

  struct plotEntryType
  {
    plotEntryType(double minMuPlusEn, double maxMuPlusEn, double minMuMinusEn, double maxMuMinusEn,		  
		  unsigned numBinsRadDivMuonEn, double minRadDivMuonEn, double maxRadDivMuonEn)
      : minMuPlusEn_(minMuPlusEn),
	maxMuPlusEn_(maxMuPlusEn),
	minMuMinusEn_(minMuMinusEn),
	maxMuMinusEn_(maxMuMinusEn),	
	numBinsRadDivMuonEn_(numBinsRadDivMuonEn),
	minRadDivMuonEn_(minRadDivMuonEn),
	maxRadDivMuonEn_(maxRadDivMuonEn),
	histogram_(0)
    {}
    ~plotEntryType() {}

    void bookHistograms(TFileDirectory& dir)
    {
      std::string histogramName = "genMuonRadCorr";
      if      ( minMuPlusEn_  < 0. && maxMuPlusEn_  < 0. ) histogramName.append("");
      else if (                       maxMuPlusEn_  < 0. ) histogramName.append(Form("_muPlusEnGt%1.0f", minMuPlusEn_));
      else if ( minMuPlusEn_  < 0.                       ) histogramName.append(Form("_muPlusEnLt%1.0f", maxMuPlusEn_));
      else                                                 histogramName.append(Form("_muPlusEn%1.0fto%1.0f", minMuPlusEn_, maxMuPlusEn_));
      if      ( minMuMinusEn_ < 0. && maxMuMinusEn_ < 0. ) histogramName.append("");
      else if (                       maxMuMinusEn_ < 0. ) histogramName.append(Form("_muMinusEnGt%1.0f", minMuMinusEn_));
      else if ( minMuMinusEn_ < 0.                       ) histogramName.append(Form("_muMinusEnLt%1.0f", maxMuMinusEn_));
      else                                                 histogramName.append(Form("_muMinusEn%1.0fto%1.0f", minMuMinusEn_, maxMuMinusEn_));
      histogram_ = dir.make<TH2D>(histogramName.data(), histogramName.data(), numBinsRadDivMuonEn_, minRadDivMuonEn_, maxRadDivMuonEn_, numBinsRadDivMuonEn_, minRadDivMuonEn_, maxRadDivMuonEn_);
    }
    void fillHistograms(double muPlusEn, double muPlusRad, double muMinusEn, double muMinusRad, double evtWeight)
    {
      if ( (minMuPlusEn_  <= 0. || muPlusEn  > minMuPlusEn_  ) &&
	   (maxMuPlusEn_  <= 0. || muPlusEn  < maxMuPlusEn_  ) &&
	   (minMuMinusEn_ <= 0. || muMinusEn > minMuMinusEn_ ) &&
	   (maxMuMinusEn_ <= 0. || muMinusEn < maxMuMinusEn_ ) ) {
	histogram_->Fill(muPlusRad, muMinusRad, evtWeight);
      }
    }

    double minMuPlusEn_;
    double maxMuPlusEn_;
    double minMuMinusEn_;
    double maxMuMinusEn_;
    unsigned numBinsRadDivMuonEn_;
    double minRadDivMuonEn_;
    double maxRadDivMuonEn_;
    
    TH2* histogram_;
  };
  std::vector<plotEntryType*> plotEntries_;

  double beamEnergy_;

  GenMuonRadiationAlgorithm* muonRadiationAlgo_;

  int verbosity_;

  int numWarnings_;
};

#endif
