// -*- C++ -*-
//
// Package:    KsAnalyzer
// Class:      KsAnalyzer
// 
/**\class KsAnalyzer KsAnalyzer.cc TrackPropagation/KsAnalyzer/src/KsAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lorenzo Viliani,32 3-B06,+41227676396,
//         Created:  Wed Jun 25 17:21:46 CEST 2014
// $Id$
//
//


// system include files
#include <memory>
#include <cmath>

#include "TH1F.h"
#include "TProfile.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

class KsAnalyzer : public edm::EDAnalyzer {
   public:
      explicit KsAnalyzer(const edm::ParameterSet&);
      ~KsAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

//      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
//      virtual void endRun(edm::Run const&, edm::EventSetup const&);
//      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
//      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
      edm::InputTag vertexTags_;
      TH1F *h_KsMass;
      TProfile *p_KsMassEta, *p_KsMassEtaLead, *p_KsMassEtaSubl, *p_KsMassPhi, *p_KsMassPhiLead, *p_KsMassPhiSubl;
      double dEtaMaxCut;
      bool doPhi; 
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
KsAnalyzer::KsAnalyzer(const edm::ParameterSet& iConfig)
  :
   vertexTags_(iConfig.getUntrackedParameter<edm::InputTag>("vertex"))

{
   //now do what ever initialization is needed
   dEtaMaxCut = iConfig.getUntrackedParameter<double>("dEtaMaxCut");
   doPhi = iConfig.getUntrackedParameter<bool>("doPhi");
 
   edm::Service<TFileService> fs;

   if(doPhi){
     p_KsMassPhi = fs->make<TProfile>("KsMassPhi", "Ks Mass vs Phi", 12, -3.14, 3.14, 0.4, 0.6);
     p_KsMassPhiLead = fs->make<TProfile>("KsMassPhiLead", "Ks Mass vs Leading Pion Phi", 12, -3.14, 3.14, 0.4, 0.6);
     p_KsMassPhiSubl = fs->make<TProfile>("KsMassPhiSubl", "Ks Mass vs Subleading Pion Phi", 12, -3.14, 3.14, 0.4, 0.6);
  
     p_KsMassPhi->SetMarkerColor(kRed);
     p_KsMassPhi->SetMarkerStyle(20);
     p_KsMassPhi->SetLineColor(kRed);
     p_KsMassPhi->GetYaxis()->SetTitle("K_{s} mass (GeV)");
     p_KsMassPhi->GetXaxis()->SetTitle("K_{s} #phi");

     p_KsMassPhiLead->SetMarkerColor(kRed);
     p_KsMassPhiLead->SetMarkerStyle(20);
     p_KsMassPhiLead->SetLineColor(kRed);
     p_KsMassPhiLead->GetYaxis()->SetTitle("K_{s} mass (GeV)");
     p_KsMassPhiLead->GetXaxis()->SetTitle("K_{s} #phi");

     p_KsMassPhiSubl->SetMarkerColor(kRed);
     p_KsMassPhiSubl->SetMarkerStyle(20);
     p_KsMassPhiSubl->SetLineColor(kRed);
     p_KsMassPhiSubl->GetYaxis()->SetTitle("K_{s} mass (GeV)");
     p_KsMassPhiSubl->GetXaxis()->SetTitle("K_{s} #phi");
  }

  else{
    h_KsMass = fs->make<TH1F>("KsMass","Ks Mass", 100, 0.4, 0.6);
    p_KsMassEta = fs->make<TProfile>("KsMassEta", "Ks Mass vs Eta", 20, -2.5, 2.5, 0.4, 0.6);
    p_KsMassEtaLead = fs->make<TProfile>("KsMassEtaLead", "Ks Mass vs Leading Pion Eta", 20, -2.5, 2.5, 0.4, 0.6);
    p_KsMassEtaSubl = fs->make<TProfile>("KsMassEtaSubl", "Ks Mass vs Subleading Pion Eta", 20, -2.5, 2.5, 0.4, 0.6);

    p_KsMassEta->SetMarkerColor(kRed);
    p_KsMassEta->SetMarkerStyle(20);
    p_KsMassEta->SetLineColor(kRed);
    p_KsMassEta->GetYaxis()->SetTitle("K_{s} mass (GeV)");
    p_KsMassEta->GetXaxis()->SetTitle("K_{s} #eta");

    p_KsMassEtaLead->SetMarkerColor(kRed);
    p_KsMassEtaLead->SetMarkerStyle(20);
    p_KsMassEtaLead->SetLineColor(kRed);
    p_KsMassEtaLead->GetYaxis()->SetTitle("K_{s} mass (GeV)");
    p_KsMassEtaLead->GetXaxis()->SetTitle("K_{s} #eta");

    p_KsMassEtaSubl->SetMarkerColor(kRed);
    p_KsMassEtaSubl->SetMarkerStyle(20);
    p_KsMassEtaSubl->SetLineColor(kRed);
    p_KsMassEtaSubl->GetYaxis()->SetTitle("K_{s} mass (GeV)");
    p_KsMassEtaSubl->GetXaxis()->SetTitle("K_{s} #eta");
  }
}

KsAnalyzer::~KsAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
KsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
   using namespace edm;
   using reco::VertexCompositeCandidateCollection;
   using reco::VertexCollection;

   Handle<VertexCollection> primaryVertex;
   iEvent.getByLabel("offlinePrimaryVertices", primaryVertex);

   reco::VertexCollection::const_iterator itPrimVertex = primaryVertex->begin();
   double pvx = itPrimVertex->x();
   double pvy = itPrimVertex->y();
   //double pvz = itPrimVertex->z();

   Handle<VertexCompositeCandidateCollection> vertex;
   iEvent.getByLabel(vertexTags_, vertex);

   for(VertexCompositeCandidateCollection::const_iterator itVertex = vertex->begin();
       itVertex != vertex->end();
       ++itVertex ) {
       
       //std::cout << "VertexCompositeCandidateCollection vertex pt = " << itVertex->pt() << std::endl;
       //std::cout << "############" << std::endl; 

       double pT_pi1 = itVertex->CompositeCandidate::daughter(0)->pt();
       double pT_pi2 = itVertex->CompositeCandidate::daughter(1)->pt();
  
       if(doPhi){

         double phi_pi1 = itVertex->CompositeCandidate::daughter(0)->phi();
         double phi_pi2 = itVertex->CompositeCandidate::daughter(1)->phi();
         double phi_lead = ( pT_pi1 >= pT_pi2 ) ? phi_pi1 : phi_pi2;
         double phi_subl = ( pT_pi1 >= pT_pi2 ) ? phi_pi2 : phi_pi1;
         double Kphi = itVertex->phi();

         double Kvx = itVertex->vx();
         double Kvy = itVertex->vy();
        // double Kvz = itVertex->vz();
         double KpT = itVertex->pt();
        // double Kpx = itVertex->px();
        // double Kpy = itVertex->py();
        // double Kpz = itVertex->pz();

         double Lx = Kvx - pvx;
         double Ly = Kvy - pvy;

         double dxy = fabs( -Lx*sin(Kphi) + Ly*cos(Kphi) );
         double Lxy = sqrt( pow(Lx,2) + pow(Ly,2) );
       
         if( KpT > 0.5 && Lxy > 0.3 && dxy < 0.1 ){
 
           p_KsMassPhi->Fill( Kphi, itVertex->mass() );
           p_KsMassPhiLead->Fill(phi_lead, itVertex->mass());
           p_KsMassPhiSubl->Fill(phi_subl, itVertex->mass());
         }            


       }
      
       else{  
         double eta_pi1 = itVertex->CompositeCandidate::daughter(0)->eta();
         double eta_pi2 = itVertex->CompositeCandidate::daughter(1)->eta();

         double dEta = fabs(eta_pi1 - eta_pi2);

         double eta_lead = ( pT_pi1 >= pT_pi2 ) ? eta_pi1 : eta_pi2;
         double eta_subl = ( pT_pi1 >= pT_pi2 ) ? eta_pi2 : eta_pi1;

         double Kvx = itVertex->vx();
         double Kvy = itVertex->vy();
        // double Kvz = itVertex->vz();
         double KpT = itVertex->pt();
        // double Kpx = itVertex->px();
        // double Kpy = itVertex->py();
        // double Kpz = itVertex->pz();
         double Kphi = itVertex->phi();

         double Lx = Kvx - pvx;
         double Ly = Kvy - pvy;

         double dxy = fabs( -Lx*sin(Kphi) + Ly*cos(Kphi) );
         double Lxy = sqrt( pow(Lx,2) + pow(Ly,2) );
       
         if( KpT > 0.5 && Lxy > 0.3 && dxy < 0.1 && dEta < dEtaMaxCut ){
           h_KsMass->Fill( itVertex->mass() );
           p_KsMassEta->Fill( itVertex->eta(), itVertex->mass() );
           p_KsMassEtaLead->Fill(eta_lead, itVertex->mass());
           p_KsMassEtaSubl->Fill(eta_subl, itVertex->mass());
	   //std::cout << "Ks Mass = " << itVertex->mass() << " GeV" << std::endl;
         }
      }
   }


#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
KsAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
KsAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
//void 
//KsAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
//{
//}

// ------------ method called when ending the processing of a run  ------------
//void 
//KsAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
//{
//}

// ------------ method called when starting to processes a luminosity block  ------------
//void 
//KsAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
//{
//}

// ------------ method called when ending the processing of a luminosity block  ------------
//void 
//KsAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
//{
//}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
KsAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(KsAnalyzer);
