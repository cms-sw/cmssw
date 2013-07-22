	// -*- C++ -*-
//
// Package:    TimeAnalyzer
// Class:      TimeAnalyzer
// 
/**\class TimeAnalyzer TimeAnalyzer.cc TimeEventStudies/TimeAnalyzer/plugins/TimeAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ivana Kurecic / Gianluca Cerminara / Giovanni Franzoni
//         Created:  Mon, 01 Jul 2013 07:44:21 GMT
// $Id$
//
//


// system include files
#include <memory>

// ==> user include files
// =-> defaults
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// =-> for development
#include <TMath.h>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TPad.h"
#include "HepMC/SimpleVector.h"
#include "THStack.h"
#include "THistPainter.h"
#include "TH2.h"

#define pi 3.141592653589793

//
// class declaration
//

class TimeAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TimeAnalyzer(const edm::ParameterSet&);
      ~TimeAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

  // pointers to the histograms for persistency
//  TH1F* h_nVtx_;
//  TH1F* h_pType_;
//  TH1F* h_pTypeSel_;
//  TH1F* h_nSCEB_;
//  TH1F* h_nSCEE_;
//  TH1F* h_masses_;
//  TH1F* h_pr_;
  TH1F* h_pperp_;
  TH1F* h_eta_;
  TH1F* h_phi_;
//  THStack* h_SCstack_;
  TH1F* h_eclusterE_;
  TH1F* h_etaclusterE_;
  TH1F* h_phiclusterE_;
  TH1F* h_eclusterB_;
  TH1F* h_etaclusterB_;
  TH1F* h_phiclusterB_; 
  TH2F* h_eta_phi_;
  THStack* h_etacstack_;
  THStack* h_phicstack_;
  TH2F* h_ceta_phi_;
  TH1F* h_delta_e;
  TH1F* h_delta_b;
  TH1F* h_energies_;
  THStack* h_ecstack_;
  TH1F* h_massofmother_;
  TH1F* h_mothertype_;


  // list of particle tipe(s) which are used for the study and matched to clusters/recHits
  std::vector<int> acceptedParticleTypes_;
  double lowestenergy_;
  std::vector<int> acceptedParentTypes_;
  float inv_nofproducts_;

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
TimeAnalyzer::TimeAnalyzer(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  
  // list of particle tipe(s) which are used for the study and matched to clusters/recHits
  acceptedParticleTypes_ = iConfig.getParameter< std::vector<int> >("acceptedParticleTypes");
  lowestenergy_ = iConfig.getParameter<double>("lowestenergy");
  acceptedParentTypes_= iConfig.getParameter<std::vector<int>>("acceptedParentTypes");
  inv_nofproducts_=1.0/acceptedParticleTypes_.size();
}


TimeAnalyzer::~TimeAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
TimeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   //   using namespace reco;
   //   using namespace std;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif



unsigned int brojrazl=0;
std::vector< std::vector<float>> particlemomenta(4, std::vector<float> (acceptedParticleTypes_.size()+1,0.));
float squaremomenta=0.;


   // reconstructed vertex in the event
   Handle<reco::VertexCollection> vertexHandle;
   iEvent.getByLabel("offlinePrimaryVerticesWithBS", vertexHandle);
   const reco::VertexCollection * vertexCollection = vertexHandle.product();
   // reco vertex multiplicity
//   h_nVtx_ -> Fill( vertexCollection->size() );
   math::XYZPoint recoVtx(0.,0.,0.);
   if (vertexCollection->size()>0) { recoVtx = vertexCollection->begin()->position();  }


   // get hold of the MC product and loop over truth-level particles 
   // standard numeric ParticleId:
   // http://www.physics.ox.ac.uk/CDF/Mphys/old/notes/pythia_codeListing.html
   Handle< edm::HepMCProduct > hepProd ;
   iEvent.getByLabel("generator",hepProd) ;
   const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();
   
   // loop over MC particles
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) 
     {
       
       // You only want particles with status 1
       if  ((*p)->status()!=1 ) continue; 
       
//       h_pType_ ->  Fill((*p)->pdg_id()); 
       
       // match only to truth of desired particle types
       if (std::find(acceptedParticleTypes_.begin(), acceptedParticleTypes_.end(), (*p)->pdg_id())==acceptedParticleTypes_.end() )  continue;
       if(lowestenergy_>(*p)->momentum().e()) continue;
       HepMC::GenParticle* mother=0;
       HepMC::GenParticle* motheraux=0;
       if ( (*p)->production_vertex()&&(*p)->production_vertex()->particles_begin(HepMC::parents)!=(*p)->production_vertex()->particles_end(HepMC::parents) ) {
	motheraux=*((*p)->production_vertex()->particles_begin(HepMC::parents));
	mother=*(motheraux->production_vertex()->particles_begin(HepMC::parents));
       }
       if(std::find(acceptedParentTypes_.begin(), acceptedParentTypes_.end(), mother->pdg_id())==acceptedParentTypes_.end() ) continue;
//Here I assume that if a parent produces one relevant particle, it will also parent the rest. Could be done in other ways, this one seems simplest for now.
       h_mothertype_-> Fill(mother->pdg_id(),inv_nofproducts_);

	for(unsigned int broj=0;broj<acceptedParticleTypes_.size();broj++) if((*p)->pdg_id()==acceptedParticleTypes_[broj]) {
		if(particlemomenta[3][broj]<(*p)->momentum().e()) {
			particlemomenta[3][broj]=(*p)->momentum().e();
			particlemomenta[0][broj]=(*p)->momentum().px();
			particlemomenta[1][broj]=(*p)->momentum().py();
			particlemomenta[2][broj]=(*p)->momentum().pz();
			}
		}

//       h_pTypeSel_ ->  Fill((*p)->pdg_id()); 

//       h_masses_ -> Fill((*p)->momentum().m());
//       h_pr_ -> Fill((*p)->momentum().pseudoRapidity());
       h_pperp_ -> Fill((*p)->momentum().perp());
       h_eta_ -> Fill((*p)->momentum().eta());
       h_phi_ -> Fill((*p)->momentum().phi());
       h_eta_phi_ -> Fill((*p)->momentum().phi(),(*p)->momentum().eta());
       h_energies_ -> Fill((*p)->momentum().e());
       float p_eta=(*p)->momentum().eta();
       float p_phi=(*p)->momentum().phi();


   // superclusters are groups of neighboring Electromagnetic Calorimeter (ECAL) recHits
   // collecting the energy relesed by (at least) one particle in the ECAL
   Handle<std::vector<reco::SuperCluster> > barrelSCHandle;
   iEvent.getByLabel("correctedHybridSuperClusters","",barrelSCHandle);
   const reco::SuperClusterCollection * barrelSCCollection = barrelSCHandle.product();
//   h_nSCEB_ -> Fill( barrelSCCollection->size() );
   for(reco::SuperClusterCollection::const_iterator blah = barrelSCCollection->begin(); blah != barrelSCCollection->end(); blah++) {
		float deltaphib=blah->phi()-p_phi; if(deltaphib<-pi) deltaphib+=2*pi; if(deltaphib>pi) deltaphib-=2*pi;
		float deltaetab=blah->eta()-p_eta;
		float deltab = sqrt(deltaphib*deltaphib+deltaetab*deltaetab);
		if(deltab<0.2) { h_delta_b -> Fill(deltab);
                h_eclusterB_ -> Fill( blah->rawEnergy() );
                h_etaclusterB_ -> Fill( blah->eta() );
                h_phiclusterB_ -> Fill( blah->phi() );
                h_ceta_phi_ -> Fill( blah->phi(), blah->eta() );
			}
                }

   Handle<std::vector<reco::SuperCluster> > endcapSCHandle;
   iEvent.getByLabel("correctedMulti5x5SuperClustersWithPreshower","",endcapSCHandle);
   const reco::SuperClusterCollection * endcapSCCollection = endcapSCHandle.product();
//   h_nSCEE_ -> Fill( endcapSCCollection->size() );
   for(reco::SuperClusterCollection::const_iterator blah = endcapSCCollection->begin(); blah != endcapSCCollection->end(); blah++) {
		float deltaphie=blah->phi()-p_phi; if(deltaphie<-pi) deltaphie+=2*pi; if(deltaphie>pi) deltaphie-=2*pi;
                float deltaetae=blah->eta()-p_eta;
                float deltae = sqrt(deltaphie*deltaphie+deltaetae*deltaetae);
                if(deltae<0.2) { h_delta_e -> Fill(deltae);
                h_eclusterE_ -> Fill( blah->rawEnergy() );
                h_etaclusterE_ -> Fill( blah->eta() );
                h_phiclusterE_ -> Fill( blah->phi() );
                h_ceta_phi_ -> Fill( blah->phi(), blah->eta() );
			}
                }


   }


	for(unsigned int brojac=0;brojac<acceptedParticleTypes_.size();brojac++) {
		if(particlemomenta[3][brojac]!=0.0) brojrazl++;
		}
	if(brojrazl==acceptedParticleTypes_.size()) {
		for(unsigned int bro=0;bro<acceptedParticleTypes_.size();bro++) {
			for(unsigned int tt=0;tt<3;tt++) particlemomenta[tt][brojrazl]+=particlemomenta[tt][bro];
			particlemomenta[3][brojrazl]+=particlemomenta[3][bro];
			}
		}
	for(unsigned int ff=0;ff<3;ff++) squaremomenta+=particlemomenta[ff][brojrazl]*particlemomenta[ff][brojrazl];
	float ugh=sqrt(particlemomenta[3][brojrazl]*particlemomenta[3][brojrazl]-squaremomenta);
	if(particlemomenta[3][brojrazl]>2.) h_massofmother_ -> Fill( ugh );



   // superclusters are groups of neighboring Electromagnetic Calorimeter (ECAL) recHits
   // collecting the energy relesed by (at least) one particle in the ECAL
//   Handle<std::vector<reco::SuperCluster> > barrelSCHandle;
//   iEvent.getByLabel("correctedHybridSuperClusters","",barrelSCHandle);
//   const reco::SuperClusterCollection * barrelSCCollection = barrelSCHandle.product();
//   h_nSCEB_ -> Fill( barrelSCCollection->size() );
//   for(reco::SuperClusterCollection::const_iterator blah = barrelSCCollection->begin(); blah != barrelSCCollection->end(); blah++) {
//		h_eclusterB_ -> Fill( blah->rawEnergy() );
//		h_etaclusterB_ -> Fill( blah->eta() );
//		h_phiclusterB_ -> Fill( blah->phi() );
//		h_ceta_phi_ -> Fill( blah->phi(), blah->eta() );
//		}
   
//   Handle<std::vector<reco::SuperCluster> > endcapSCHandle;
//   iEvent.getByLabel("correctedMulti5x5SuperClustersWithPreshower","",endcapSCHandle);
//   const reco::SuperClusterCollection * endcapSCCollection = endcapSCHandle.product();
//   h_nSCEE_ -> Fill( endcapSCCollection->size() );
//   for(reco::SuperClusterCollection::const_iterator blah = endcapSCCollection->begin(); blah != endcapSCCollection->end(); blah++) {
//                h_eclusterE_ -> Fill( blah->rawEnergy() );
//                h_etaclusterE_ -> Fill( blah->eta() );
//                h_phiclusterE_ -> Fill( blah->phi() );
//		h_ceta_phi_ -> Fill( blah->phi(), blah->eta() );
//                }


   // for an example of matching between truth MC particles and ECAL superclusters see: 
   // http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/Minnesota/Hgg/ClusteringWithPU/plugins/SCwithTruthPUAnalysis.cc?revision=1.3&view=markup 



}


// ------------ method called once each job just before starting event loop  ------------
void 
TimeAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;
  TFileDirectory subDir=fs->mkdir("baseHistoDir");  
  


  // histograms need be booked in the beginJob, which is run only once at  the  beginning of execution
//  h_nVtx_     = fs->make<TH1F>("h_nVtx","no. of primary vertices; num vertices reco",10,0.,10.);
//  h_pType_    = fs->make<TH1F>("h_pType","truth particle type; truth particle type",80,-40.,40.);
//  pTypeSel_ = fs->make<TH1F>("h_pTypeSel","truth particle type (selected); truth particle type",80,-40.,40.);
//  h_nSCEE_    = fs->make<TH1F>("h_nSCEE","number superclusters in EE; num EE SC  ",100,0.,10.);
//  h_nSCEB_    = fs->make<TH1F>("h_nSCEB","number superclusters in EB; num EB SC",100,0.,10.);
//  h_masses_     = fs->make<TH1F>("h_mase","mass",100,0.50995e-3,0.51005e-3);
//  h_pr_       = fs->make<TH1F>("h_pr","pseudorapidity",100,-10.,10.);
  h_pperp_ = fs->make<TH1F>("h_pperp","transverse momentum",100,0.,90.);
  h_eta_      = fs->make<TH1F>("h_eta","eta",100,-3.,3.);
  h_phi_      = fs->make<TH1F>("h_phi","phi",100,-3.5,3.5);
//  h_SCstack_ = fs->make<THStack>("SCstack","superclusters total");
  h_eclusterE_ = fs->make<TH1F>("h_eclusterE","cluster energy, E",100,0.,500.);
  h_etaclusterE_ = fs->make<TH1F>("h_etaclusterE","eta E",100,-3.,3.);
  h_phiclusterE_ = fs->make<TH1F>("h_phiclusterE","phi E",100,-3.5,3.5);
  h_eclusterB_ = fs->make<TH1F>("h_eclusterB","cluster energy, B",100,0.,500.);
  h_etaclusterB_ = fs->make<TH1F>("h_etaclusterB","eta B",100,-3.,3.);
  h_phiclusterB_ = fs->make<TH1F>("h_phiclusterB","phi B",100,-3.5,3.5);
  h_eta_phi_  = fs->make<TH2F>("h_eta_phi","eta vs phi",100,-3.5,3.5,100,-10.,10.);
  h_etacstack_= fs->make<THStack>("h_etacstack","E&B cluster eta stack");
  h_phicstack_= fs->make<THStack>("h_phicstack","E&B cluster phi stack");
  h_ceta_phi_ = fs->make<TH2F>("h_ceta_phi","E&B cluster eta vs phi",100,3.5,3.5,100,-3.,3.);
  h_delta_e    = fs->make<TH1F>("h_delta","deltaE",100,0.,0.2);
  h_delta_b    = fs->make<TH1F>("h_delta","deltaB",100,0.,0.2);
  h_energies_ = fs->make<TH1F>("h_enegies","energies",100,0.,500.);
  h_ecstack_  = fs->make<THStack>("h_ecstack","E&B energy stack");
  h_massofmother_ = fs->make<TH1F>("h_massofmother","e+e- particle energy",100,0.,160.);
  h_mothertype_ = fs->make<TH1F>("h_mothertype_","type of mother",80,-40.,40.);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
TimeAnalyzer::endJob() 
{

//h_SCstack_ -> Add(h_nSCEE_);
//h_SCstack_ -> Add(h_nSCEB_);
h_eta_phi_ -> SetOption("colz");
h_etacstack_ -> Add(h_etaclusterE_); h_etacstack_ -> Add(h_etaclusterB_);
h_phicstack_ -> Add(h_phiclusterB_); h_phicstack_ -> Add(h_phiclusterE_);
h_ceta_phi_ -> SetOption("colz");
h_ecstack_ -> Add(h_eclusterE_); h_ecstack_ -> Add(h_eclusterB_);

}

// ------------ method called when starting to processes a run  ------------
/*
void 
TimeAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
TimeAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
TimeAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
TimeAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TimeAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TimeAnalyzer);

