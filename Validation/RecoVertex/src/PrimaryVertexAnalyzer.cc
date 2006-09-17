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
 
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


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
  h1_vtx_chi2_  = new TH1F("vtxchi2","chi squared",100,0.,1000.);
  h1_vtx_ndf_ = new TH1F("vtxndf","degrees of freedom",100,0.,100.);
  h1_tklinks_ = new TH1F("tklinks","Usable track links",2,-0.5,1.5);
  h1_nans_ = new TH1F("nans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
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
  h1_tklinks_->Write();
  h1_nans_->Write();
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
      	      << v->position().x() << " " << v->covariance(0, 0) << " " 
      	      << v->position().y() << " " << v->covariance(1, 1) << " " 
      	      << v->position().z() << " " << v->covariance(2, 2) << " " 
	      << std::endl;

    try {
      for(reco::track_iterator t = v->tracks_begin(); 
	  t!=v->tracks_end(); t++) {
	// illegal charge
        if ( (**t).charge() < -1 || (**t).charge() > 1 ) {
	  h1_tklinks_->Fill(0.);
        }
        else {
	  h1_tklinks_->Fill(1.);
        }
      }
    }
    catch (...) {
      // exception thrown when trying to use linked track
      h1_tklinks_->Fill(0.);
    }

    h1_nbvtx_in_event_->Fill(recVtxs->size()*1.);
    h1_nbtks_in_vtx_->Fill(v->tracksSize());
    h1_resx_->Fill(v->position().x());
    h1_resy_->Fill(v->position().y());
    h1_resz_->Fill(v->position().z());
    h1_pullx_->Fill(v->position().x()/v->xError());
    h1_pully_->Fill(v->position().y()/v->yError());
    h1_pullz_->Fill(v->position().z()/v->zError());
    h1_vtx_chi2_->Fill(v->chi2());
    h1_vtx_ndf_->Fill(v->ndof());

    bool problem = false;
    h1_nans_->Fill(1.,isnan(v->position().x())*1.);
    h1_nans_->Fill(2.,isnan(v->position().y())*1.);
    h1_nans_->Fill(3.,isnan(v->position().z())*1.);

    int index = 3;
    for (int i = 0; i != 3; i++) {
      for (int j = i; j != 3; j++) {
	index++;
	h1_nans_->Fill(index*1., isnan(v->covariance(i, j))*1.);
	if (isnan(v->covariance(i, j))) problem = true;
	// in addition, diagonal element must be positive
	if (j == i && v->covariance(i, j) < 0) {
	  h1_nans_->Fill(index*1., 1.);
	  problem = true;
	}
      }
    }

    if (problem) {
      // analyze track parameter covariance definiteness
      double data[25];
      try {
	int itk = 0;
	for(reco::track_iterator t = v->tracks_begin(); 
	    t!=v->tracks_end(); t++) {
	  int i2 = 0;
	  for (int i = 0; i != 5; i++) {
	    for (int j = 0; j != 5; j++) {
	      data[i2] = (**t).covariance(i, j);
	      i2++;
	    }
	  }
	  gsl_matrix_view m 
	    = gsl_matrix_view_array (data, 5, 5);
	  
	  gsl_vector *eval = gsl_vector_alloc (5);
	  gsl_matrix *evec = gsl_matrix_alloc (5, 5);
	  
	  gsl_eigen_symmv_workspace * w = 
	    gsl_eigen_symmv_alloc (5);
	  
	  gsl_eigen_symmv (&m.matrix, eval, evec, w);
	  
	  gsl_eigen_symmv_free (w);
	  
	  gsl_eigen_symmv_sort (eval, evec, 
				GSL_EIGEN_SORT_ABS_ASC);
	  
	  // print sorted eigenvalues
	  {
	    std::cout << "Track " << itk++ << std::endl;
	    int i;
	    for (i = 0; i < 5; i++) {
	      double eval_i 
		= gsl_vector_get (eval, i);
	      gsl_vector_view evec_i 
		= gsl_matrix_column (evec, i);
	      
	      printf ("eigenvalue = %g\n", eval_i);
	      //	      printf ("eigenvector = \n");
	      //	      gsl_vector_fprintf (stdout, 
	      //				  &evec_i.vector, "%g");
	    }
	  }
	}
      }
      catch (...) {
	// exception thrown when trying to use linked track
	break;
      }
    }
  }
}
