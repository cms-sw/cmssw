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

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TFile.h"
#include "TH1.h"


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
  TH1F* h_nVtx;
  TH1F* h_pType;

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


   // reconstructed vertex in the event
   Handle<reco::VertexCollection> vertexHandle;
   iEvent.getByLabel("offlinePrimaryVerticesWithBS", vertexHandle);
   const reco::VertexCollection * vertexCollection = vertexHandle.product();
   // reco vertex multiplicity
   h_nVtx -> Fill( vertexCollection->size() );
   math::XYZPoint recoVtx(0.,0.,0.);
   if (vertexCollection->size()>0) recoVtx = vertexCollection->begin()->position();
   

   // get hold of the MC product and loop over truth-level particles 
   // standard numeric ParticleId: http://www.physics.ox.ac.uk/CDF/Mphys/old/notes/pythia_codeListing.html
   Handle< edm::HepMCProduct > hepProd ;
   iEvent.getByLabel("generator",hepProd) ;
   const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();
   
   
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {
     // match only to truth of electrons or photons
     h_pType ->  Fill((*p)->pdg_id());
     if ( !( (fabs((*p)->pdg_id())==11 || (*p)->pdg_id()==22) && (*p)->status()==1 )  )  continue;
    
     //std::cout << "particle found SCwithTruthPUAnalysis " <<  (*p)->pdg_id() << "\t" << (*p)->status() << std::endl;
   }
   



}


// ------------ method called once each job just before starting event loop  ------------
void 
TimeAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;
  TFileDirectory subDir=fs->mkdir("baseHistoDir");  

  // histograms need be booked in the beginJob, which is run only once at  the  beginning of execution
  h_nVtx  = fs->make<TH1F>("h_nVtx","no. of primary vertices; num vertices reco",40,0.,40.);
  h_pType = fs->make<TH1F>("h_pType","truth particle type",81,-40.,40.);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
TimeAnalyzer::endJob() 
{
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
