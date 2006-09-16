#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// vertex stuff
#include "DataFormats/VertexReco/interface/Vertex.h"
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
  h1_nbvtx_in_event_ = new TH1F("nbvtx","nb vertices in event",100,0.,100.);
  h1_nbtks_in_vtx_ = new TH1F("nbtksinvtx","nb tracks in vertex",100,0.,100.); 
  h1_resx_  = new TH1F("resx","residual x",100,-0.1,0.1);
  h1_resy_ = new TH1F("resy","residual y",100,-0.1,0.1);
  h1_resz_ = new TH1F("resz","residual z",100,-0.1,0.1);
  h1_pullx_ = new TH1F("pullx","pull x",100,-25.,25.);
  h1_pully_ = new TH1F("pully","pull y",100,-25.,25.);
  h1_pullz_ = new TH1F("pullz","pull z",100,-25.,25.);
  h1_vtx_chi2_  = new TH1F("vtxchi2", "chisqu",100,0.,1000.);
  h1_vtx_ndf_ = new TH1F("vtxndf", "ndf",100,0.,100.);
}


void PrimaryVertexAnalyzer::endJob() {
  rootFile_->cd();
  h1_nbvtx_in_event_->Write();
  h1_nbtks_in_vtx_->Write();
  h1_resx_->Write();
  h1_resy_->Write();
  h1_resz_->Write();
  h1_pullx_->Write();
  h1_pully_->Write();
  h1_pullz_->Write();
  h1_vtx_chi2_->Write();
  h1_vtx_ndf_->Write();
}


// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel("offlinePrimaryVerticesFromCTFTracks", 
		    recVtxs);
  std::cout << "vertices " << recVtxs->size() << std::endl;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    std::cout << "recvtx " 
              << v->tracksSize() << " "
	      << v->chi2() << " " 
	      << v->ndof() << " " 
	      << v->position().x() << " " << v->position().x()/sqrt(v->error(0,0)) << " " 
	      << v->position().y() << " " << v->position().y()/sqrt(v->error(1,1)) << " " 
	      << v->position().z() << " " << v->position().z()/sqrt(v->error(2,2)) << " " 
	      << std::endl;

    int ok=1;
    for ( reco::track_iterator t = v->tracks_begin(); t!=v->tracks_end(); t++ )
	  {
      if ( (**t).charge() < -1 || (**t).charge() > 1 )
	    {
        std::cout << "Error: illegal track charge " << (**t).charge()
                  << "!" << std::endl;
        ok=0;
        break;
      }
    }

    std::cout << "[OVAL] see if vertex track links work: " << ok << std::endl;

    h1_nbvtx_in_event_->Fill(recVtxs->size()*1.);
    h1_nbtks_in_vtx_->Fill(v->tracksSize());
    h1_resx_->Fill(v->position().x());
    h1_resy_->Fill(v->position().y());
    h1_resz_->Fill(v->position().z());
    h1_pullx_->Fill(v->position().x()/sqrt(v->error(0,0)));
    h1_pully_->Fill(v->position().y()/sqrt(v->error(1,1)));
    h1_pullz_->Fill(v->position().z()/sqrt(v->error(2,2)));
    h1_vtx_chi2_->Fill(v->chi2());
    h1_vtx_ndf_->Fill(v->ndof());
  }


}

