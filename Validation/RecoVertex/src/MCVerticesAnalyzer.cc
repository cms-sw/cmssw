// -*- C++ -*-
//
// Package:    MCVerticesAnalyzer
// Class:      MCVerticesAnalyzer
// 
/**\class MCVerticesAnalyzer MCVerticesAnalyzer.cc TrackingPFG/PileUp/src/MCVerticesAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
//
//


// system include files
#include <memory>
#include <numeric>

#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"


#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

//
// class decleration
//


class MCVerticesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MCVerticesAnalyzer(const edm::ParameterSet&);
  ~MCVerticesAnalyzer();
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
      // ----------member data ---------------------------

  

  const bool m_useweight;

  edm::EDGetTokenT< double > m_doubleToken;
  edm::EDGetTokenT< std::vector<PileupSummaryInfo> > m_vecPileupSummaryInfoToken;
  edm::EDGetTokenT< edm::HepMCProduct > m_hepMCProductToken;

  TH1F* m_hnvtx;
  TH2F* m_hnvtxvsbx;
  TH1F* m_hlumi;
  TH2F* m_hnvtxvslumi;
  TH1F* m_hnvtxweight;
  TProfile* m_hnvtxweightprof;
  TH1F* m_hmainvtxx;
  TH1F* m_hmainvtxy;
  TH1F* m_hmainvtxz;
  TH1F* m_hpileupvtxz;

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
MCVerticesAnalyzer::MCVerticesAnalyzer(const edm::ParameterSet& iConfig)
  : m_useweight( iConfig.getParameter< bool >( "useWeight" ) )
  , m_doubleToken( consumes< double >( iConfig.getParameter< edm::InputTag >( "weightProduct" ) ) )
  , m_vecPileupSummaryInfoToken( consumes< std::vector<PileupSummaryInfo> >( iConfig.getParameter< edm::InputTag >( "pileupSummaryCollection" ) ) )
  , m_hepMCProductToken( consumes< edm::HepMCProduct >( iConfig.getParameter< edm::InputTag >( "mcTruthCollection" ) ) )
{
   //now do what ever initialization is needed



  usesResource("TFileService");
  edm::Service<TFileService> tfserv;

  m_hnvtx = tfserv->make<TH1F>("nvtx","Number of pileup vertices",60,-0.5,59.5);
  m_hnvtx->GetXaxis()->SetTitle("Number of Interactions");

  m_hnvtxvsbx = tfserv->make<TH2F>("nvtxvsbx","Number of pileup vertices vs BX",9,-4.5,4.5,60,-0.5,59.5);
  m_hnvtxvsbx->GetXaxis()->SetTitle("BX number");
  m_hnvtxvsbx->GetYaxis()->SetTitle("Number of Interactions");

  m_hlumi = tfserv->make<TH1F>("lumi","BX luminosity*xsect",200,0.,50.);
  m_hlumi->GetXaxis()->SetTitle("Average Number of Interactions");

  m_hnvtxvslumi = tfserv->make<TH2F>("nvtxvslumi","Npileup vs BX luminosity*xsect",200,0.,50.,60,-0.5,59.5);
  m_hnvtxvslumi->GetXaxis()->SetTitle("Average Number of Interactions");  m_hnvtxvslumi->GetYaxis()->SetTitle("Number of Interactions");

  if(m_useweight) {
    m_hnvtxweight = tfserv->make<TH1F>("nvtxweight","Number of pileup vertices (1-w)",60,-0.5,59.5);
    m_hnvtxweight->GetXaxis()->SetTitle("Number of Interactions");
    m_hnvtxweightprof = tfserv->make<TProfile>("nvtxweightprof","Mean (1-w) vs Number of pileup interactions",60,-0.5,59.5);
    m_hnvtxweightprof->GetXaxis()->SetTitle("Number of Interactions");
  }

  m_hmainvtxx = tfserv->make<TH1F>("mainvtxx","Main vertex x position",200,-.5,.5);
  m_hmainvtxx->GetXaxis()->SetTitle("X (cm)");
  m_hmainvtxy = tfserv->make<TH1F>("mainvtxy","Main vertex y position",200,-.5,.5);
  m_hmainvtxy->GetXaxis()->SetTitle("Y (cm)");
  m_hmainvtxz = tfserv->make<TH1F>("mainvtxz","Main vertex z position",600,-30.,30.);
  m_hmainvtxz->GetXaxis()->SetTitle("Z (cm)");
  m_hpileupvtxz = tfserv->make<TH1F>("pileupvtxz","PileUp vertices z position",600,-30.,30.);
  m_hpileupvtxz->GetXaxis()->SetTitle("Z (cm)");

}


MCVerticesAnalyzer::~MCVerticesAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MCVerticesAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   double weight = 1.;

   if(m_useweight) {
     edm::Handle<double> weightprod;
     iEvent.getByToken( m_doubleToken, weightprod );

     weight = *weightprod;

   }


   edm::Handle<std::vector<PileupSummaryInfo> >  pileupinfos;
   iEvent.getByToken( m_vecPileupSummaryInfoToken, pileupinfos );

   //

   if(pileupinfos.isValid()) {

  // look for the intime PileupSummaryInfo

     std::vector<PileupSummaryInfo>::const_iterator pileupinfo;

     for(pileupinfo = pileupinfos->begin(); pileupinfo != pileupinfos->end() ; ++pileupinfo) {
       m_hnvtxvsbx->Fill(pileupinfo->getBunchCrossing(),pileupinfo->getPU_NumInteractions(),weight);
     } 


     for(pileupinfo = pileupinfos->begin(); pileupinfo != pileupinfos->end() ; ++pileupinfo) {
       if(pileupinfo->getBunchCrossing()==0) break;
     } 
 
     //

     if(pileupinfo->getBunchCrossing()!=0) {
       edm::LogError("NoInTimePileUpInfo") << "Cannot find the in-time pileup info " << pileupinfo->getBunchCrossing();
     }
     else {

       m_hlumi->Fill(pileupinfo->getTrueNumInteractions(),weight);
       m_hnvtx->Fill(pileupinfo->getPU_NumInteractions(),weight);
       m_hnvtxvslumi->Fill(pileupinfo->getTrueNumInteractions(),pileupinfo->getPU_NumInteractions(),weight);

       if(m_useweight) {
	 m_hnvtxweight->Fill(pileupinfo->getPU_NumInteractions(),1.-weight);
	 m_hnvtxweightprof->Fill(pileupinfo->getPU_NumInteractions(),1.-weight);
       }
       
       const std::vector<float>& zpositions = pileupinfo->getPU_zpositions();
       
       for(std::vector<float>::const_iterator zpos = zpositions.begin() ; zpos != zpositions.end() ; ++zpos) {
	 
	 m_hpileupvtxz->Fill(*zpos,weight);
	 
       }
     }
   }
   // main interaction part

   edm::Handle< edm::HepMCProduct > EvtHandle ;
   iEvent.getByToken( m_hepMCProductToken, EvtHandle );

   if(EvtHandle.isValid()) {

     const HepMC::GenEvent* Evt = EvtHandle->GetEvent();

   // get the first vertex
     
     if(Evt->vertices_begin() != Evt->vertices_end()) {

       m_hmainvtxx->Fill((*Evt->vertices_begin())->point3d().x()/10.,weight);
       m_hmainvtxy->Fill((*Evt->vertices_begin())->point3d().y()/10.,weight);
       m_hmainvtxz->Fill((*Evt->vertices_begin())->point3d().z()/10.,weight);

     }
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCVerticesAnalyzer);
