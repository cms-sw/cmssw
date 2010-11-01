/** \class TopLeptonTauFilter
 *
 * Top DiLepton (e,mu,tau) skimming
 * default pt thresholds (lepton and jets) set to 15 GeV
 * default eta thresholds (lepton and jets) set to 3
 * At least two leptons and two jets present for each channel
 *
 * $Date: 2010/03/25 09:22:19 $
 * $Revision: 1.13 $
 *
 * \author Michele Gallinaro and Nuno Almeida - LIP
 *
 */

#include <iostream>
#include <string>
#include <list>
#include <cmath>
#include <cstdio>
#include <vector>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"    

#include "TopQuarkAnalysis/TopSkimming/plugins/TopLeptonTauFilter.h"

//electron includes
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

//muon includes
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
//tau includes
#include "DataFormats/TauReco/interface/BaseTau.h"
//jet includes
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

class PtSorter {
public:
  template <class T> bool operator() ( const T& a, const T& b ) {
    return ( a.pt() > b.pt() );
  }
};

TopLeptonTauFilter::TopLeptonTauFilter( const edm::ParameterSet& iConfig ) :
  nEvents_(0), nAccepted_(0)
{

  ElecFilter_ = iConfig.getParameter<bool>( "ElecFilter" );
  MuonFilter_ = iConfig.getParameter<bool>( "MuonFilter" );
  TauFilter_  = iConfig.getParameter<bool>( "TauFilter" );
  JetFilter_  = iConfig.getParameter<bool>( "JetFilter" );
   
  Elecsrc_    = iConfig.getParameter<edm::InputTag>( "Elecsrc" );
  Muonsrc_    = iConfig.getParameter<edm::InputTag>( "Muonsrc" );
  Tausrc_     = iConfig.getParameter<edm::InputTag>( "Tausrc" );
  CaloJetsrc_ = iConfig.getParameter<edm::InputTag>( "CaloJetsrc" );
  
  NminMuon_     = iConfig.getParameter<int>( "NminMuon" );
  NminTau_      = iConfig.getParameter<int>( "NminTau" );
  NminElec_     = iConfig.getParameter<int>( "NminElec" );
  NminCaloJet_  = iConfig.getParameter<int>( "NminCaloJet" );
  
  ElecPtmin_    = iConfig.getParameter<double>( "ElecPtmin" );
  MuonPtmin_    = iConfig.getParameter<double>( "MuonPtmin" );
  TauPtmin_     = iConfig.getParameter<double>( "TauLeadTkPtmin" );
  CaloJetPtmin_ = iConfig.getParameter<double>( "CaloJetPtmin" );
  
}

/*------------------------------------------------------------------------*/

TopLeptonTauFilter::~TopLeptonTauFilter() 
{}

/*------------------------------------------------------------------------*/

bool TopLeptonTauFilter::filter( edm::Event& iEvent, 
				const edm::EventSetup& iSetup )
{

  bool filterResult(true);
  nEvents_++;

  if(ElecFilter_){ filterResult = electronFilter(iEvent,iSetup); }
  if(MuonFilter_){ filterResult = muonFilter(iEvent,iSetup); }
  if(TauFilter_) { filterResult = tauFilter(iEvent,iSetup); }
  if(JetFilter_) { filterResult = jetFilter(iEvent,iSetup); }
  
  if (filterResult) nAccepted_++;
  
  return filterResult;
}


bool TopLeptonTauFilter::electronFilter( edm::Event& iEvent, const edm::EventSetup
& iSetup ){

  // dealing with electrons
  edm::Handle<reco::GsfElectronCollection> ElecHandle;
  iEvent.getByLabel( Elecsrc_, ElecHandle );
  if ( ElecHandle->empty() && NminElec_!=0 ) return false;
  reco::GsfElectronCollection TheElecs = *ElecHandle;
  std::stable_sort( TheElecs.begin(), TheElecs.end(), PtSorter() );
  
  int nElec = 0;
  for ( reco::GsfElectronCollection::const_iterator it = TheElecs.begin();
	it != TheElecs.end(); it++ ) {
    if ( (it->pt() > ElecPtmin_) 
	 && (fabs(it->eta()) < 3.0) ) {
       nElec++;
    }
  }
  if ( nElec < NminElec_ ) return false;
  
     
  return true;
}

bool TopLeptonTauFilter::muonFilter( edm::Event& iEvent, const edm::EventSetup
& iSetup ){

  edm::Handle<reco::MuonCollection> MuonHandle;
  iEvent.getByLabel( Muonsrc_, MuonHandle );
  if ( MuonHandle->empty() && NminMuon_!=0 ) return false;
  reco::MuonCollection TheMuons = *MuonHandle;
  std::stable_sort( TheMuons.begin(), TheMuons.end(), PtSorter() );
  
  int nMuon = 0;

  
  for ( reco::MuonCollection::const_iterator it = TheMuons.begin();
	it != TheMuons.end(); it++ ) {
    if ( (it->pt() > MuonPtmin_) 
	 && (fabs(it->eta()) < 3.0) ) {
      nMuon++;
    }
  }
  
  if ( nMuon < NminMuon_ ) return false;
    
  nAccepted_++;

  return true;
}



bool TopLeptonTauFilter::tauFilter( edm::Event& iEvent, const edm::EventSetup
& iSetup ){

  edm::Handle<reco::BaseTauCollection> TauHandle;
  iEvent.getByLabel(Tausrc_,TauHandle);
  const reco::BaseTauCollection& myTauCollection=*(TauHandle.product());
  if ( myTauCollection.empty() && NminTau_!=0 ) return false;
  
  int nTau = 0;
  for(reco::BaseTauCollection::const_iterator it =myTauCollection.begin();it !=myTauCollection.end();it++)
    {
      reco::TrackRef theLeadTk = it->leadTrack();
      if(!theLeadTk) {}
      else{
        double leadTkPt  = (*theLeadTk).pt();
	double leadTkEta = (*theLeadTk).eta();

        if ( (leadTkPt> TauPtmin_)
	 && (fabs(leadTkEta) < 2.4) ) {
          nTau++;
	}
      }
    }
  if ( nTau < NminTau_ ) return false;
       
  return true;
}




bool TopLeptonTauFilter::jetFilter( edm::Event& iEvent, const edm::EventSetup
& iSetup ){

  edm::Handle<reco::CaloJetCollection> CaloJetsHandle;
  iEvent.getByLabel( CaloJetsrc_, CaloJetsHandle );
  if ( CaloJetsHandle->empty() && NminCaloJet_!=0 ) return false;

  int nJet = 0;
  for ( reco::CaloJetCollection::const_iterator it = CaloJetsHandle->begin(); 
	it != CaloJetsHandle->end(); it++ ) {
    if ( (fabs(it->eta()) < 3.0) &&
	 (it->pt() > CaloJetPtmin_) ) nJet++;
  }
  
  if ( nJet < NminCaloJet_ ) return false;
  
  return true;

}




/*------------------------------------------------------------------------*/

void TopLeptonTauFilter::endJob()
{

//  edm::LogVerbatim( "TopLeptonTauFilter" ) 
   edm::LogInfo( "TopLeptonTauFilter" )
    << "\n Events read " << nEvents_
    << " Events accepted " << nAccepted_
    << "\nEfficiency " << (double)(nAccepted_)/(double)(nEvents_) 
    << std::endl;

}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(TopLeptonTauFilter);
