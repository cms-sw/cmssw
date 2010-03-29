#ifndef TopLeptonTauFilter_h
#define TopLeptonTauFilter_h

/** \class TopLeptonTauFilter
 *
 * Top DiLepton (e,mu,tau) skimming
 * default pt thresholds (lepton and jets) set to 15 GeV 
 * default eta thresholds (lepton and jets) set to 3
 * At least two leptons and two jets present for each channel
 *
 * $Date: 2007/08/07 10:50:13 $
 * $Revision: 1.1 $
 *
 * \author Michele Gallinaro and Nuno Almeida - LIP
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"    

class TopLeptonTauFilter : public edm::EDFilter 
{
 public:
  explicit TopLeptonTauFilter( const edm::ParameterSet& );
  ~TopLeptonTauFilter();
  virtual bool filter( edm::Event&, const edm::EventSetup& );
  virtual void endJob();

 private:
 
  bool electronFilter( edm::Event&, const edm::EventSetup& );
  bool muonFilter( edm::Event&, const edm::EventSetup& );
  bool jetFilter( edm::Event&, const edm::EventSetup& );
  bool tauFilter( edm::Event&, const edm::EventSetup& );
 
  edm::InputTag Elecsrc_; 
  edm::InputTag Muonsrc_;  
  edm::InputTag Tausrc_;
  edm::InputTag CaloJetsrc_;
  
  int NminElec_, NminMuon_, NminTau_, NminCaloJet_;
  double ElecPtmin_,MuonPtmin_,TauPtmin_,CaloJetPtmin_ ;
  
  bool ElecFilter_,MuonFilter_,TauFilter_, JetFilter_;
  
  unsigned int nEvents_;
  unsigned int nAccepted_;
};

#endif
