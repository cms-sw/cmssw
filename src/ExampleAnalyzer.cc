// -*- C++ -*-
//
// Package:    ExampleAnalyzer
// Class:      ExampleAnalyzer
// 
/**\class ExampleAnalyzer ExampleAnalyzer.cc ExampleProducer/ExampleAnalyzer/src/ExampleAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Frank GOLF
//         Created:  Wed Aug 19 18:07:32 CEST 2009
// $Id: ExampleAnalyzer.cc,v 1.1 2009/08/25 12:41:33 fgolf Exp $
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// include these files to access the MET collections
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/Common/interface/ValueMap.h"

//
// class decleration
//

class ExampleAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ExampleAnalyzer(const edm::ParameterSet&);
      ~ExampleAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  edm::InputTag caloMETlabel;        // input label for raw calo MET 
  edm::InputTag tcMETlabel;	     // input label for track-correct MET
  edm::InputTag muCorrMETlabel;	     // input label for caloMET corrected for muons
  edm::InputTag pfMETlabel;	     // input label for particle-flow MET
  edm::InputTag muJESCorrMETlabel;   // input label for caloMET corrected for muons and JES
  edm::InputTag muonLabel;           // input label for muon collection
  edm::InputTag muValueMaplabel;     // input label for the muon correction value map
  edm::InputTag tcMETValueMaplabel;  // input label for the tcMET value map 
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ExampleAnalyzer::ExampleAnalyzer(const edm::ParameterSet& iConfig)

{
  caloMETlabel       = iConfig.getParameter<edm::InputTag>("caloMETlabel_"      );     
  tcMETlabel         = iConfig.getParameter<edm::InputTag>("tcMETlabel_"        );       
  muCorrMETlabel     = iConfig.getParameter<edm::InputTag>("muCorrMETlabel_"    );   
  pfMETlabel         = iConfig.getParameter<edm::InputTag>("pfMETlabel_"        );       
  muJESCorrMETlabel  = iConfig.getParameter<edm::InputTag>("muJESCorrMETlabel_" );
  muonLabel          = iConfig.getParameter<edm::InputTag>("muonLabel_"         );
  muValueMaplabel    = iConfig.getParameter<edm::InputTag>("muValueMaplabel_"   );
  tcMETValueMaplabel = iConfig.getParameter<edm::InputTag>("tcMETValueMaplabel_");
}


ExampleAnalyzer::~ExampleAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ExampleAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get handles to collections

  edm::Handle< edm::View<reco::CaloMET> > caloMEThandle;
  iEvent.getByLabel(caloMETlabel, caloMEThandle);
  
  edm::Handle< edm::View<reco::CaloMET> > muCorrMEThandle;
  iEvent.getByLabel(muCorrMETlabel, muCorrMEThandle);

  edm::Handle< edm::View<reco::CaloMET> > muJESCorrMEThandle;
  iEvent.getByLabel(muJESCorrMETlabel, muJESCorrMEThandle);

  edm::Handle< edm::View<reco::MET> >     tcMEThandle;
  iEvent.getByLabel(tcMETlabel, tcMEThandle);

  edm::Handle< edm::View<reco::PFMET> >   pfMEThandle;
  iEvent.getByLabel(pfMETlabel, pfMEThandle);

  edm::Handle< reco::MuonCollection > muon_h;
  iEvent.getByLabel(muonLabel, muon_h);

  edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > muVMhandle;
  iEvent.getByLabel(muValueMaplabel, muVMhandle);

  edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMETVMhandle;
  iEvent.getByLabel(tcMETValueMaplabel, tcMETVMhandle);

  // get MET and METphi from objects

  /*
  float caloMET         = (caloMEThandle->front() ).et();
  float caloMETphi      = (caloMEThandle->front() ).phi();

  float muCorrMET       = (muCorrMEThandle->front() ).et();
  float muCorrMETphi    = (muCorrMEThandle->front() ).phi();
  			  
  float muJESCorrMET    = (muJESCorrMEThandle->front() ).et();
  float muJESCorrMETphi = (muJESCorrMEThandle->front() ).phi();

  float tcMET           = (tcMEThandle->front() ).et();
  float tcMETphi        = (tcMEThandle->front() ).phi();
			  
  float pfMET           = (pfMEThandle->front() ).et();
  float pfMETphi        = (pfMEThandle->front() ).phi();
  */


  const unsigned int nMuons = muon_h->size();

  edm::ValueMap<reco::MuonMETCorrectionData> muVM = *muVMhandle;

  std::vector<bool>  mu_flag;
  std::vector<float> mu_x;
  std::vector<float> mu_y;

  // loop over ValueMap entries and extract flag and x, y components (one for each muon)

  for( unsigned int mus = 0; mus < nMuons; mus++ ) {

    reco::MuonRef muref( muon_h, mus);
    reco::MuonMETCorrectionData muCorrData = (muVM)[muref];

    mu_flag.push_back( muCorrData.type() );
    mu_x.push_back( muCorrData.corrX() );
    mu_y.push_back( muCorrData.corrY() );

    std::cout << "flag = " << mu_flag[mus] << "\n";
    std::cout << "mu_x = " << mu_x[mus]    << "\n";
    std::cout << "mu_y = " << mu_y[mus]    << "\n";
  }

  edm::ValueMap<reco::MuonMETCorrectionData> tcVM = *tcMETVMhandle;

  std::vector<bool>  tc_flag;
  std::vector<float> tc_x;
  std::vector<float> tc_y;

  // same thing for tcMET

  for( unsigned int mus = 0; mus < nMuons; mus++ ) {

    reco::MuonRef muref( muon_h, mus);
    reco::MuonMETCorrectionData muCorrData = (muVM)[muref];

    tc_flag.push_back( muCorrData.type() );
    tc_x.push_back( muCorrData.corrX() );
    tc_y.push_back( muCorrData.corrY() );

    std::cout << "flag = " << mu_flag[mus] << "\n";
    std::cout << "mu_x = " << mu_x[mus]    << "\n";
    std::cout << "mu_y = " << mu_y[mus]    << "\n";
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
ExampleAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ExampleAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ExampleAnalyzer);
