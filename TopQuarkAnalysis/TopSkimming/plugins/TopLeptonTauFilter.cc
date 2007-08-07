/** \class TopLeptonTauFilter
 *
 * Top DiLepton (e,mu,tau) skimming
 * default pt thresholds (lepton and jets) set to 15 GeV
 * default eta thresholds (lepton and jets) set to 3
 * At least two leptons and two jets present for each channel
 *
 * $Date: 2007/07/12 09:17:22 $
 * $Revision: 1.1 $
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
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

//muon includes
#include "DataFormats/MuonReco/interface/Muon.h"
//tau includes
#include "DataFormats/TauReco/interface/Tau.h"
//jet includes
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"


using namespace edm;
using namespace std;
using namespace reco;

class PtSorter {
public:
  template <class T> bool operator() ( const T& a, const T& b ) {
    return ( a.pt() > b.pt() );
  }
};

TopLeptonTauFilter::TopLeptonTauFilter( const edm::ParameterSet& iConfig ) :
  nEvents_(0), nAccepted_(0)
{
  
  ElecFilter_ = iConfig.getUntrackedParameter<bool>( "ElecFilter" );
  MuonFilter_ = iConfig.getUntrackedParameter<bool>( "MuonFilter" );
  TauFilter_  = iConfig.getUntrackedParameter<bool>( "TauFilter" );
  JetFilter_  = iConfig.getUntrackedParameter<bool>( "JetFilter" );
   
  Elecsrc_    = iConfig.getParameter<InputTag>( "Elecsrc" );
  Muonsrc_    = iConfig.getParameter<InputTag>( "Muonsrc" );
  Tausrc_     = iConfig.getParameter<InputTag>( "Tausrc" );
  CaloJetsrc_ = iConfig.getParameter<InputTag>( "CaloJetsrc" );
  
  NminMuon_      = iConfig.getUntrackedParameter<int>( "NminMuon", 1 );
  NminTau_      = iConfig.getUntrackedParameter<int>( "NminTau", 1 );
  NminElec_     = iConfig.getUntrackedParameter<int>( "NminElec", 1 );
  NminCaloJet_  = iConfig.getUntrackedParameter<int>( "NminCaloJet", 2 );
  
  ElecPtmin_    = iConfig.getUntrackedParameter<double>( "ElecPtmin", 15. );
  MuonPtmin_    = iConfig.getUntrackedParameter<double>( "MuonPtmin", 15. );
  TauPtmin_     = iConfig.getUntrackedParameter<double>( "TauLeadTkPtmin", 15. );
  CaloJetPtmin_ = iConfig.getUntrackedParameter<double>( "CaloJetPtmin", 15. );
  
  
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

  if(ElecFilter_){filterResult=electronFilter(iEvent,iSetup);}
  if(MuonFilter_){filterResult &= muonFilter(iEvent,iSetup);}
  if(TauFilter_) {filterResult &= tauFilter(iEvent,iSetup); }
  if(JetFilter_) {filterResult &= jetFilter(iEvent,iSetup); }
  
  if (filterResult) nAccepted_++;
  
  return filterResult;
}


bool TopLeptonTauFilter::electronFilter( edm::Event& iEvent, const edm::EventSetup
& iSetup ){

  // dealing with electrons
  Handle<PixelMatchGsfElectronCollection> ElecHandle;
  try {
    iEvent.getByLabel( Elecsrc_, ElecHandle );
  }
  catch ( cms::Exception& ex ) {
    edm::LogError( "TopLeptonTauFilter" ) 
      << "Unable to get Elec collection "
      << Elecsrc_.label();
    return false;
  }
  if ( ElecHandle->empty() && NminElec_!=0 ) return false;
  
  PixelMatchGsfElectronCollection TheElecs = *ElecHandle;
  std::stable_sort( TheElecs.begin(), TheElecs.end(), PtSorter() );
  
  int nElec = 0;
  for ( PixelMatchGsfElectronCollection::const_iterator it = TheElecs.begin();
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

Handle<MuonCollection> MuonHandle;
  try {
    iEvent.getByLabel( Muonsrc_, MuonHandle );
  }
  catch ( cms::Exception& ex ) {
    edm::LogError( "TopLeptonTauFilter" ) 
      << "Unable to get Muon collection "
      << Muonsrc_.label();
    return false;
  }
  if ( MuonHandle->empty() && NminMuon_!=0 ) return false;
  MuonCollection TheMuons = *MuonHandle;
  std::stable_sort( TheMuons.begin(), TheMuons.end(), PtSorter() );
  
  int nMuon = 0;

  
  for ( MuonCollection::const_iterator it = TheMuons.begin();
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

  Handle<TauCollection> TauHandle;
  try {
  iEvent.getByLabel(Tausrc_,TauHandle);
  }
  catch ( cms::Exception& ex ) {
    edm::LogError( "TopLeptonTauFilter" ) 
      << "Unable to get Tau collection "
      << Tausrc_.label();
    return false;
  }
  const TauCollection& myTauCollection=*(TauHandle.product());
  
  if ( myTauCollection.empty() && NminTau_!=0 ) return false;
  
  int nTau = 0;
  for(TauCollection::const_iterator it =myTauCollection.begin();it !=myTauCollection.end();it++)
    {
      TrackRef theLeadTk = it->getLeadingTrack();
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
 Handle<CaloJetCollection> CaloJetsHandle;

  try {
    iEvent.getByLabel( CaloJetsrc_, CaloJetsHandle );
  } 
  catch ( cms::Exception& ex ) {
    edm::LogError( "TopLeptonTauFilter" ) 
      << "Unable to get CaloJet collection "
      << CaloJetsrc_.label();
    return false;
  }

  if ( CaloJetsHandle->empty() && NminCaloJet_!=0 ) return false;

  int nJet = 0;
  for ( CaloJetCollection::const_iterator it = CaloJetsHandle->begin(); 
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

   cout << "\n Events read " << nEvents_
    << " Events accepted " << nAccepted_
    << "\nEfficiency " << (double)(nAccepted_)/(double)(nEvents_) 
    << endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TopDiLeptonFilter);
