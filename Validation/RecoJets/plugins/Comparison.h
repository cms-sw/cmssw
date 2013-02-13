#ifndef Comparison_h
#define Comparison_h

#include <memory>
#include <string>
#include <vector>

#include "TH1F.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Validation/RecoJets/plugins/CaloJetQualifier.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/RecoJets/interface/NameScheme.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
class Comparison {
  
 public:
  Comparison(const edm::ParameterSet&);
  ~Comparison(){};
  std::map<unsigned int, unsigned int> operator()(const Ref&, const Rec&);
  void book();
  void book(ofstream&);
  void summarize();
  
 private:

  double maxDR_;
  double minPtRef_, maxPtRef_;
  double minPtRec_, maxPtRec_;
  double minEtaRef_, maxEtaRef_;
  double minEtaRec_, maxEtaRec_;

 private:

  std::vector<TH1F*> hDR_;   // deltaR matching
  std::vector<TH1F*> hPt_;   // matching efficiency vs pt
  std::vector<TH1F*> hEta_;  // matching efficiency vs eta 

  Alg alg_;
  RefQualifier refQualifier_;
  RecQualifier recQualifier_; 

  unsigned int found_;
  unsigned int missed_;
  unsigned int failed_; 
};

template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
Comparison<Ref, RefQualifier, Rec, RecQualifier, Alg>::Comparison(const edm::ParameterSet& cfg):
  maxDR_( cfg.getParameter<double>( "maxDR" ) ),
  minPtRef_ ( cfg.getParameter<double>( "minPtRef"  ) ),
  maxPtRef_ ( cfg.getParameter<double>( "maxPtRef"  ) ),
  minPtRec_ ( cfg.getParameter<double>( "minPtRec"  ) ),
  maxPtRec_ ( cfg.getParameter<double>( "maxPtRec"  ) ),
  minEtaRef_( cfg.getParameter<double>( "minEtaRef" ) ),
  maxEtaRef_( cfg.getParameter<double>( "maxEtaRef" ) ),
  minEtaRec_( cfg.getParameter<double>( "minEtaRec" ) ),
  maxEtaRec_( cfg.getParameter<double>( "maxEtaRec" ) ),
  refQualifier_( cfg ), recQualifier_( cfg ),
  found_( 0 ), missed_( 0 ), failed_( 0 )
{
  std::string hist=cfg.getParameter<std::string>("hist");
  if( hist.empty() )
    book();
  else{
    ofstream file(hist.c_str(), std::ios::out);
    book(file);
  }
}

template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
std::map<unsigned int, unsigned int> 
Comparison<Ref, RefQualifier, Rec, RecQualifier, Alg>::operator()(const Ref& refs, const Rec& recs)
{
  int refIdx=0;
  std::map<unsigned int, unsigned int> matches;
  for(typename Ref::const_iterator ref=refs.begin(); 
      ref!=refs.end(); ++ref, ++refIdx ){
    if( !(minEtaRef_<ref->eta() && ref->eta()<maxEtaRef_) ) 
      // retrict to visible range in eta
      continue;
    
    if( !(minPtRef_ <ref->pt()  && ref->pt() <maxPtRef_ ) )
      // restrict to visible range in pt
      continue;

    if( !refQualifier_( *ref ) ) 
      // restrtict to properly qualified reference object
      continue;

    int jetIdx=0;
    int match=-1;
    double dist=-1.;    
    for(typename Rec::const_iterator rec = recs.begin();
	rec!=recs.end(); ++rec, ++jetIdx ){
      if( !(minEtaRec_<rec->eta() && rec->eta()<maxEtaRec_) ) 
	// retrict to visible range in eta
	continue;
      
      if( !(minPtRec_ <rec->pt()  && rec->pt() <maxPtRec_ ) )
	// restrict to visible range in pt
	continue;
      
      if( !recQualifier_( *rec ) ) 
	// restrtict to properly qualified CaloJet
	continue;
      
      double dR = alg_(*ref, *rec);
      if( dist<0 || dR<dist ){
	dist  = dR;
	match = jetIdx;
      }
    }
    if( match<0 ) ++failed_;
    if( match>=0 ){
      if(hDR_ .size()>0) hDR_ [0]->Fill( dist );
      if(hPt_ .size()>0) hPt_ [0]->Fill( ref->pt()  );
      if(hEta_.size()>0) hEta_[0]->Fill( ref->eta() );
    }
    if( 0<dist && dist<maxDR_ ){
      ++found_;
      if( match>=0 ){
	if(hDR_ .size()>1) hDR_ [1]->Fill( dist );
	if(hPt_ .size()>1) hPt_ [1]->Fill( ref->pt()  );
	if(hEta_.size()>1) hEta_[1]->Fill( ref->eta() );
      }
      if( !matches.insert(std::pair<int, int>(refIdx, match)).second )
	edm::LogWarning ( "MapMismatch" ) 
	  << "Match could not be inserted in map; entry already exited?!";
    }
    else ++missed_;
  }
  return matches;
}

template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
void Comparison<Ref, RefQualifier, Rec, RecQualifier, Alg>::book()
{
  edm::Service<TFileService> fs;
  if( !fs )
    throw edm::Exception( edm::errors::Configuration, "TFile Service is not registered in cfg file" );

  NameScheme match("match");
  static const unsigned int MAXHIST=2;
  for(unsigned int idx=0; idx<MAXHIST; ++idx){
    hDR_ .push_back( fs->make<TH1F>( match.name(      "deltaR", idx), match.name("dR"), 100, 0.,   1.) );
    hPt_ .push_back( fs->make<TH1F>( match.name(      "effPt",  idx), match.name("pt"),  30, 0., 300.) );
    hEta_.push_back( fs->make<TH1F>( match.name(      "effEta", idx), match.name("eta"), 30,-3.,   3.) );
  }
}

template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
void Comparison<Ref, RefQualifier, Rec, RecQualifier, Alg>::book(ofstream& file)
{
  edm::Service<TFileService> fs;
  if( !fs )
    throw edm::Exception( edm::errors::Configuration, "TFile Service is not registered in cfg file" );

  NameScheme match("match");
  static const unsigned int MAXHIST=2;
  for(unsigned int idx=0; idx<MAXHIST; ++idx){
    hDR_ .push_back( fs->make<TH1F>( match.name(file, "deltaR", idx), match.name("dR"), 100, 0.,   1.) );
    hPt_ .push_back( fs->make<TH1F>( match.name(file, "effPt",  idx), match.name("pt"),  30, 0., 300.) );
    hEta_.push_back( fs->make<TH1F>( match.name(file, "effEta", idx), match.name("eta"), 30,-3.,   3.) );
  }
}

template <typename Ref, typename RefQualifier, typename Rec, typename RecQualifier, typename Alg>
void Comparison<Ref, RefQualifier, Rec, RecQualifier, Alg>::summarize()
{
  unsigned int all=found_+missed_+failed_;
  if(all>0){
    edm::LogInfo("MatchSummary") << "=============================================";
    edm::LogInfo("MatchSummary") << "Reference :";
    edm::LogInfo("MatchSummary") << "CaloJet   :";
    edm::LogInfo("MatchSummary") << "fraction of found  jets: " << 100*found_ /all << "%";
    edm::LogInfo("MatchSummary") << "fraction of missed jets: " << 100*missed_/all << "%";
    edm::LogInfo("MatchSummary") << "fraction of failed jets: " << 100*failed_/all << "%";
  }
  else{
    edm::LogWarning ( "MatchOrBalanceFault" ) 
      << "No missed, failed nor counts found";    
  }
}

#endif
