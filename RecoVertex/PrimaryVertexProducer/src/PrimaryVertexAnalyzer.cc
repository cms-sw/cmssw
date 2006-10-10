#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexAnalyzer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// vertex stuff
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// Root
#include <TH1.h>
#include <TFile.h>



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexAnalyzer::PrimaryVertexAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  // open output file to store histograms}
  outputFile_   = iConfig.getUntrackedParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); 
}


PrimaryVertexAnalyzer::~PrimaryVertexAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete rootFile_;
}



//
// member functions
//
void PrimaryVertexAnalyzer::beginJob(edm::EventSetup const&){
  rootFile_->cd();
  h1_pullx_ = new TH1F("pullx","pull x",100,-25.,25.);
  h1_pully_ = new TH1F("pully","pull y",100,-25.,25.);
  h1_pullz_ = new TH1F("pullz","pull z",100,-25.,25.);
  h1_chi2_  = new TH1F("chi2", "chisqu",100,0.,1000.);
}


void PrimaryVertexAnalyzer::endJob() {
  rootFile_->cd();
  h1_pullx_->Write();
  h1_pully_->Write();
  h1_pullz_->Write();
  h1_chi2_->Write();
}


// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel("offlinePrimaryVerticesFromCTFTracks", "PrimaryVertex",
		    recVtxs);
  std::cout << "vertices " << recVtxs->size() << std::endl;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    std::cout << "recvtx " 
	      << v->chi2() << " " 
	      << v->ndof() << " " 
	      << v->position().x() << " " << v->position().x()/sqrt(v->error(0,0)) << " " 
	      << v->position().y() << " " << v->position().y()/sqrt(v->error(1,1)) << " " 
	      << v->position().z() << " " << v->position().z()/sqrt(v->error(2,2)) << " " 
	      << std::endl;
    h1_pullx_->Fill(v->position().x()/sqrt(v->error(0,0)));
    h1_pully_->Fill(v->position().y()/sqrt(v->error(1,1)));
    h1_pullz_->Fill(v->position().z()/sqrt(v->error(2,2)));
    h1_chi2_->Fill(v->chi2());
  }


}

//define this as a plug-in
//DEFINE_FWK_MODULE(PrimaryVertexAnalyzer)
