// -*- C++ -*-
//
// Package:    RPCHSCP
// Class:      RPCHSCP
// 
/**\class RPCHSCP RPCHSCP.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Carrillo camilo.carrillo AT cern.ch
//         Created:  Wed Aug  6 17:45:45 CEST 2008
// $Id: RPCHSCP.cc,v 1.2 2009/05/07 11:25:01 carrillo Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"


#include "SUSYBSMAnalysis/HSCP/interface/RPCHSCPCANDIDATE.h"

//Root
#include "TFile.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"
#include "TCanvas.h"
#include <fstream>


//
// class decleration
//

class RPCHSCP : public edm::EDProducer {
   public:
      explicit RPCHSCP(const edm::ParameterSet&);
      ~RPCHSCP();
  
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::ofstream fileMatrix;
      std::string rootFileName;
      std::string fileMatrixName;
      //int matrixbitTOTAL[4][7];
      TFile* theFile;
      TH1F * matrixHisto;

      TH1F * efficiencyeta;
      TH1F * efficiencyphi;
      TH1F * efficiencybeta;

      TH1F * expectedeta;
      TH1F * expectedphi;
      TH1F * expectedbeta;

      TH1F * observedeta;
      TH1F * observedphi;
      TH1F * observedbeta;

      TH1F * residualeta;
      TH1F * residualphi;
      TH1F * residualbeta;

      int Maxbx[7];
      int totalHSCP;
      std::string fileMatrixname; 
      std::string partLabel;
      // ----------member data ---------------------------
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
RPCHSCP::RPCHSCP(const edm::ParameterSet& iConfig)
{
  partLabel = iConfig.getUntrackedParameter<std::string>("partLabel");
  fileMatrixName= iConfig.getUntrackedParameter<std::string>("fileMatrixname");
  rootFileName = iConfig.getUntrackedParameter<std::string>("rootFileName");
  produces<int>("JustATest"); 
}


RPCHSCP::~RPCHSCP(){

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RPCHSCP::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel("rpcRecHits",rpcHits);
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  std::cout<<"Reading the particles container"<<std::endl;

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel(partLabel,genParticles );
  
  std::cout << " Number of Particles in this event: " << genParticles->size() << std::endl;

  reco::GenParticleCollection::const_iterator partIt;
  for(partIt=genParticles->begin();partIt!=genParticles->end();++partIt) {
    std::cout<<" Particle Id="<<partIt->pdgId()<<std::endl;
    if(partIt->pdgId()==-2000015){
      //ABOUT ETA
      //std::cout<<"\t\t Filling the histogram Eta="<<partIt->eta()<<std::endl;
      //etaHisto->Fill(partIt->eta(),countHitsInRPC);
      //soloEtaHisto->Fill(partIt->eta());
      //if(hscp)soloEtaMyHisto->Fill(partIt->eta());
      //ABOUT BETA
      float p=partIt->p();
      float e=partIt->energy();
      float betamc=p/e;
      //betaHisto->Fill(beta);
      //if(hscp)betaMyHisto->Fill(beta);
      float pt = partIt->pt();
      //float et = partIt->et();
      float betaT=pt/e;
      //betaTHisto->Fill(betaT);
      //if(count!=0)bxLayerFile<<"\t"<<" eta="<<partIt->eta()<<" beta="<<beta<<"c";
      //if(count!=0)fileMatrix<<" eta="<<partIt->eta()<<" beta="<<beta<<"c Event "<<iEvent.id().event()<<"\n";
      //if(fabs(partIt->eta()>1.14))etaout++;
      std::cout<<"\t phimc="<<partIt->phi()<<" etamc="<<partIt->eta()<<"betamc="<<betamc<<" pmc="<<p<<"GeV ptmc="<<pt<<"GeV mmc="<<sqrt(e*e-p*p)<<"GeV"<<std::endl;
      
      int event = iEvent.id().event();
      float etamc = partIt->eta();
      float phimc = partIt->phi();
      
      RPCHSCPCANDIDATE rpcCandidate(iEvent,iSetup);

      expectedeta->Fill(etamc);
      expectedphi->Fill(phimc);
      expectedbeta->Fill(betamc);
      
      std::cout<<" rpcCandidate.found()="<<rpcCandidate.found()<<std::endl;

      if(rpcCandidate.found()){
	std::cout<<" etamc="<<etamc<<" phimc="<<phimc<<" betamc="<<betamc<<std::endl;
	std::cout<<" eta  ="<<rpcCandidate.eta()<<" phi  ="<<rpcCandidate.phi()<<" beta="<<rpcCandidate.beta()<<std::endl;

	
	float diffeta = etamc - rpcCandidate.eta();
	float diffphi = phimc - rpcCandidate.phi();
	float diffbeta = betamc - rpcCandidate.beta();

	if(fabs(diffeta)<=0.3 && fabs(diffphi)<=0.03){
	  std::cout<<"Coincidence also in direction!!!"<<std::endl;
	  observedeta->Fill(etamc);
	  observedphi->Fill(phimc);
	  observedbeta->Fill(betamc);
	  residualphi->Fill(diffphi);
	  residualeta->Fill(diffeta);
	  residualbeta->Fill(diffbeta);
	}else{
	  std::cout<<" Identified but in different direction"<<std::endl;
	}
      }else{
	std::cout<<" eta="<<etamc<<" phi="<<phimc<<std::endl;
	std::cout<<" Not identified"<<std::endl;
      }
    }
  }
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
RPCHSCP::beginJob(const edm::EventSetup&){

  fileMatrix.open(fileMatrixname.c_str());
  theFile = new TFile(rootFileName.c_str(),"RECREATE");
  //matrixHisto = new TH1F("LayersandBX","Histogram 2D Layers and BX",6,0.5,6.5,4,-0.5,3.5);
  matrixHisto = new TH1F("LayersandBX","Histogram 2D Layers and BX",30,-0.5,29.5);
  
  efficiencyeta = new TH1F("EtaEff","Eta Efficiency",100,-2.5,2.5);		 
  efficiencyphi = new TH1F("PhiEff","Phi Efficiency",100,-3.1415926,3.1415926); 
  efficiencybeta = new TH1F("BetaEff","Beta Efficiency",100,0,1);                
  
  expectedeta = new TH1F("EtaExpected","Eta Expected",100,-2.5,2.5);		 
  expectedphi = new TH1F("PhiExpected","Phi Expected",100,-3.1415926,3.1415926); 
  expectedbeta = new TH1F("BetaExpected","Beta Expected",100,0,1);                
  
  observedeta = new TH1F("EtaObserved","Eta Observed",100,-2.5,2.5);		 
  observedphi = new TH1F("PhiObserved","Phi Observed",100,-3.1415926,3.1415926); 
  observedbeta = new TH1F("BetaObserved","Beta Observed",100,0,1);                

  residualeta = new TH1F("Residual Eta","Eta Residuals",100,-0.2,0.2);
  residualphi = new TH1F("Residual Phi","Phi Residuals",100,-0.02,0.02);
  residualbeta = new TH1F("ResidualBeta","Beta Residuals",100,-3.0,3.0);
  
  totalHSCP=0;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCHSCP::endJob() {
  for(int k=1;k<=100;k++){
    float effeta = 0;
    float erreta = 0;
    if(expectedeta->GetBinContent(k)!=0){
      effeta = observedeta->GetBinContent(k)/expectedeta->GetBinContent(k);
      erreta = sqrt(effeta*(1-effeta)/expectedeta->GetBinContent(k));
    }
    efficiencyeta->SetBinContent(k,effeta);
    efficiencyeta->SetBinError(k,erreta);
    
    float effbeta = 0;
    float errbeta = 0;
    if(expectedbeta->GetBinContent(k)!=0){
      effbeta = observedbeta->GetBinContent(k)/expectedbeta->GetBinContent(k);
      errbeta = sqrt(effbeta*(1-effbeta)/expectedbeta->GetBinContent(k));
    }
    efficiencybeta->SetBinContent(k,effbeta);
    efficiencybeta->SetBinError(k,errbeta);

    float effphi = 0;
    float errphi = 0;
    if(expectedphi->GetBinContent(k)!=0){
      effphi = observedphi->GetBinContent(k)/expectedphi->GetBinContent(k);
      errphi = sqrt(effphi*(1-effphi)/expectedphi->GetBinContent(k));
    }
    efficiencyphi->SetBinContent(k,effphi);
    efficiencyphi->SetBinError(k,errphi);
  }
  
  theFile->cd();
  matrixHisto->Write();

  efficiencyeta->Write();
  efficiencyphi->Write();
  efficiencybeta->Write();

  expectedeta->Write();
  expectedphi->Write();
  expectedbeta->Write();

  observedeta->Write();
  observedphi->Write();
  observedbeta->Write();

  residualeta->Write();
  residualphi->Write();
  residualbeta->Write();

  theFile->Close();
  fileMatrix.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCHSCP);
