#ifndef CalibAnalyzer_h
#define CalibAnalyzer_h

#include <memory>
#include <string>
#include <vector>

#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Validation/RecoJets/interface/CompType.h"
#include "Validation/RecoJets/interface/NameScheme.h"


template <typename Ref, typename Rec, typename Alg>
class CalibAnalyzer : public edm::EDAnalyzer {

 public:

  explicit CalibAnalyzer(const edm::ParameterSet&);
  ~CalibAnalyzer(){};
  
 private:

  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){ alg_.summarize(); };

  void fill(const double& var, const double& val, const std::vector<double>& bins, const std::vector<TH1F*>& hists);

 private:

  edm::InputTag recs_;
  edm::InputTag refs_;
  std::string hist_;
  int type_;                                      // comparison type (0:ratio/1:rel diff)
  int bins_;                                      // number of bins in fit histograms
  double min_, max_;                              // min/max of fit histograms
  std::vector<double> binsPt_, binsEta_;          // binning in pt/eta

 private:
  
  TH2F *recVsRef_;
  TH1F *calPt_, *resPt_;                          // cal/res vs pt
  std::vector<TH1F*> ktPt_;                       // calibration plots vs pt
  TH1F *calEta_, *resEta_;                        // cal/res vs eta
  std::vector<TH1F*> ktEta_;                      // calibration plots vs eta
  std::vector<TH1F*> calEtaPt_, resEtaPt_;        // cal/res vs eta and pt
  std::vector<std::vector<TH1F*> > ktEtaPt_;      // calibration plots vs eta and pt

  Alg alg_;                                       // matching/balancing algorithm
};

template <typename Ref, typename Rec, typename Alg>
CalibAnalyzer<Ref, Rec, Alg>::CalibAnalyzer(const edm::ParameterSet& cfg):
  recs_( cfg.getParameter<edm::InputTag>( "recs" ) ),
  refs_( cfg.getParameter<edm::InputTag>( "refs" ) ),
  hist_( cfg.getParameter<std::string > ( "hist" ) ),
  type_( cfg.getParameter<int>( "type" ) ),
  bins_( cfg.getParameter<int>( "bins" ) ),
  min_ ( cfg.getParameter<double>( "min" ) ),
  max_ ( cfg.getParameter<double>( "max" ) ),
  binsPt_ ( cfg.getParameter<std::vector<double> >( "binsPt"  ) ),
  binsEta_( cfg.getParameter<std::vector<double> >( "binsEta" ) ),
  alg_( cfg )
{
}

template <typename Ref, typename Rec, typename Alg>
void CalibAnalyzer<Ref, Rec, Alg>::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<Ref> refs;
  evt.getByLabel(refs_, refs);

  edm::Handle<Rec> recs;
  evt.getByLabel(recs_, recs);
  
  // do matching
  std::map<unsigned int, unsigned int> matches=alg_(*refs, *recs);

  if( !matches.size()>0 )
    edm::LogWarning ( "NoMatchOrBalance" ) 
      << "No single match/balance found to any Rec object in collection";
  
  // fill comparison plots for matched jets
  for(std::map<unsigned int, unsigned int>::const_iterator match=matches.begin(); 
      match!=matches.end(); ++match){

    CompType cmp(type_);
    double val=cmp((*refs)[match->first].pt(), (*recs)[match->second].pt());

    fill((*refs)[match->first].pt(),  val, binsPt_,  ktPt_ );// inclusive binned in pt 
    fill((*refs)[match->first].eta(), val, binsEta_, ktEta_);// inclusive binned in eta  

    // differential in eta binned in pt   
    for(int idx=0; idx<((int)binsEta_.size()-1); ++idx)
      if( (binsEta_[idx]<(*refs)[match->first].eta()) && ((*refs)[match->first].eta()<binsEta_[idx+1]) )
	fill((*refs)[match->first].pt(),  val, binsPt_,  ktEtaPt_[idx] );
    recVsRef_->Fill( TMath::Log10((*refs)[match->first].pt()), TMath::Log10((*recs)[match->second].pt()) );
  }
}

template <typename Ref, typename Rec, typename Alg>
void CalibAnalyzer<Ref, Rec, Alg>::fill(const double& var, const double& val, const std::vector<double>& bins, const std::vector<TH1F*>& hists)
{
  for(unsigned int idx=0; idx<(bins.size()-1); ++idx){
    if( (bins[idx]<var) && (var<bins[idx+1]) ){
      hists[idx]->Fill( val );
    }
  }
}

template <typename Ref, typename Rec, typename Alg>
void CalibAnalyzer<Ref, Rec, Alg>::beginJob()
{
  if( hist_.empty() )
    return;

  edm::Service<TFileService> fs;
  if( !fs )
    throw edm::Exception( edm::errors::Configuration, "TFile Service is not registered in cfg file" );

  ofstream hist(hist_.c_str(), std::ios::out);
  NameScheme val("val"), fit("fit"), cal("cal"), res("res");

  // book additional control histograms
  recVsRef_= fs->make<TH2F>( val.name("recVsRef"),val.name("recVsRef"), 20, 1., 3., 20, 1., 3.);

  // book kt histograms differential in pt
  for(int idx=0; idx<((int)binsPt_.size()-1); ++idx)
    ktPt_.push_back( fs->make<TH1F>(fit.name(hist, "ktPt",idx), fit.name("kt",idx), bins_, min_, max_) );
  calPt_= fs->make<TH1F>(cal.name(hist, "ktPt"), cal.name("calPt"), ((int)binsPt_.size()-1), &binsPt_[0]);
  resPt_= fs->make<TH1F>(res.name(hist, "ktPt"), res.name("resPt"), ((int)binsPt_.size()-1), &binsPt_[0]);
  
  // book kt histograms differential in eta  
  for(int jdx=0; jdx<((int)binsEta_.size()-1); ++jdx)
    ktEta_.push_back(fs->make<TH1F>(fit.name(hist, "ktEta",jdx),fit.name("kt",jdx), bins_, min_, max_) );
  calEta_= fs->make<TH1F>(cal.name(hist, "ktEta"), cal.name("calEta"), ((int)binsEta_.size()-1), &binsEta_[0]);
  resEta_= fs->make<TH1F>(res.name(hist, "ktEta"), res.name("resEta"), ((int)binsEta_.size()-1), &binsEta_[0]);
  
  // book kt histograms differential in eta and pt
  for(int jdx=0; jdx<((int)binsEta_.size()-1); ++jdx){
    std::vector<TH1F*> buffer;
    calEtaPt_.push_back(fs->make<TH1F>(cal.name(hist,"ktEtaPt",jdx), cal.name("calEtaPt",jdx), ((int)binsPt_.size()-1), &binsPt_[0]));
    resEtaPt_.push_back(fs->make<TH1F>(res.name(hist,"ktEtaPt",jdx), res.name("resEtaPt",jdx), ((int)binsPt_.size()-1), &binsPt_[0]));
    for(int idx=0; idx<((int)binsPt_.size()-1); ++idx)
      buffer.push_back( fs->make<TH1F>(fit.name(hist, "ktEtaPt",jdx,idx), fit.name("ktEtaPt",jdx,idx), bins_, min_, max_) );
    ktEtaPt_.push_back(buffer); 
  }
}

#endif
