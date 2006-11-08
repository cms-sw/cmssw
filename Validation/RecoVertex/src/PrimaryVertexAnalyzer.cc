#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// reco track and vertex 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// simulated vertices,..., add <use name=SimDataFormats/Vertex> and <../Track>
#include <SimDataFormats/Vertex/interface/SimVertex.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>

//generator level + CLHEP
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenVertex.h"
// HepPDT // for simtracks
#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "SimGeneral/HepPDT/interface/HepParticleData.h"

// Root
#include <TH1.h>
#include <TH2.h>
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
  simG4_=iConfig.getParameter<edm::InputTag>( "simG4" );
  recoTrackProducer_= iConfig.getUntrackedParameter<std::string>("recoTrackProducer");
  // open output file to store histograms}
  outputFile_  = iConfig.getUntrackedParameter<std::string>("outputFile");
  vtxSample_   = iConfig.getUntrackedParameter<std::string>("vtxSample");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); 
  simUnit_= 0.1;
  verbose_= iConfig.getUntrackedParameter<bool>("verbose", false);
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
  // release validation histograms used in DoCompare.C
  h["nbvtx"]        = new TH1F("nbvtx","nb vertices in event",100,-0.5,99.5);
  h["nbtksinvtx"]   = new TH1F("nbtksinvtx","reconstructed tracks in vertex",100,-0.5,99.5); 
  h["resx"]         = new TH1F("resx","residual x",100,-0.04,0.04);
  h["resy"]         = new TH1F("resy","residual y",100,-0.04,0.04);
  h["resz"]         = new TH1F("resz","residual z",100,-0.1,0.1);
  h["pullx"]        = new TH1F("pullx","pull x",100,-25.,25.);
  h["pully"]        = new TH1F("pully","pull y",100,-25.,25.);
  h["pullz"]        = new TH1F("pullz","pull z",100,-25.,25.);
  h["vtxchi2"]      = new TH1F("vtxchi2","chi squared",100,0.,100.);
  h["vtxndf"]       = new TH1F("vtxndf","degrees of freedom",100,0.,100.);
  h["tklinks"]      = new TH1F("tklinks","Usable track links",2,-0.5,1.5);
  h["nans"]         = new TH1F("nans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
  // more histograms
  h["eff"]          = new TH1F("eff","efficiency",2, -0.5, 1.5);
  h["nbsimtksinvtx"] = new TH1F("nbsimtksinvtx","simulated tracks in vertex",100,-0.5,99.5); 
  h["xrec"]         = new TH1F("xrec","reconstructed x",100,-0.01,0.01);
  h["yrec"]         = new TH1F("yrec","reconstructed y",100,-0.01,0.01);
  h["zrec"]         = new TH1F("zrec","reconstructed z",100,-10.,10.);
  h["xsim"]         = new TH1F("xsim","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  h["ysim"]         = new TH1F("ysim","simulated y",100,-0.01,0.01);
  h["zsim"]         = new TH1F("zsim","simulated z",100,-10.,10.);
  h["nrecvtx"]      = new TH1F("nrecvtx","# of reconstructed vertices", 50, -0.5, 49.5);
  h["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  h["nrectrk"]      = new TH1F("nrectrk","# of reconstructed tracks", 50, -0.5, 49.5);
  h["nsimtrk"]      = new TH1F("nsimtrk","# of simulated tracks", 50, -0.5, 49.5);
}


void PrimaryVertexAnalyzer::endJob() {
  rootFile_->cd();
  // save all histograms created in beginJob()
  for(std::map<std::string,TH1*>::const_iterator hist=h.begin(); hist!=h.end(); hist++){
    hist->second->Write();
  }
}


// helper functions
bool PrimaryVertexAnalyzer::matchVertex(const simPrimaryVertex  &vsim, 
				       const reco::Vertex       &vrec){
  return (fabs(vsim.z*simUnit_-vrec.z())<0.0500); // =500um
}

bool PrimaryVertexAnalyzer::isResonance(const HepMC::GenParticle * p){
  return  ! HepPDT::theTable().getParticleData(p->pdg_id())->stable() 
    && HepPDT::theTable().getParticleData(p->pdg_id())->cTau()<1e-6;
}



// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel(vtxSample_, recVtxs);
  
  Handle<edm::SimVertexContainer> simVtcs;
  iEvent.getByLabel( simG4_, simVtcs);
  
  Handle<SimTrackContainer> simTrks;
  iEvent.getByLabel( simG4_, simTrks);

  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);

  Handle<HepMCProduct> evt;
  //iEvent.getByType(evt);
  iEvent.getByLabel("source",evt);
  const HepMC::GenEvent *genEvt=evt->GetEvent();
  //genEvt->print();

  /*
  if(evt->GetEvent()->signal_process_vertex()){
    HepLorentzVector vsig=evt->GetEvent()->signal_process_vertex()->position();
    std::cout <<" signal vertex " << vsig.x() << " " << vsig.z() << std::endl;
  }else{
    std::cout <<" no signal vertex"  << std::endl;
  }
  */


  if(verbose_){
    int ivtx=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      std::cout << "recvtx "<< ivtx++
	      << "#trk " << std::setw(3) << v->tracksSize()
	      << " chi2 " << std::setw(4) << v->chi2() 
	      << " ndof " << std::setw(3) << v->ndof() << std::endl 
      	      << " x "  << std::setw(6) << v->x() 
	      << " dx " << std::setw(6) << v->xError()<< std::endl
      	      << " y "  << std::setw(6) << v->y() 
 	      << " dy " << std::setw(6) << v->yError()<< std::endl
      	      << " z "  << std::setw(6) << v->z() 
 	      << " dz " << std::setw(6) << v->zError()
	      << std::endl;
    }

    int i=0;
    for(edm::SimVertexContainer::const_iterator vsim=simVtcs->begin();
	vsim!=simVtcs->end(); ++vsim){
      std::cout << i++ << ")" 
		<< " sim x=" << vsim->position().x()*simUnit_
		<< " sim y=" << vsim->position().y()*simUnit_
		<< " sim z=" << vsim->position().z()*simUnit_
		<< " sim t=" << vsim->position().t()
		<< " parent=" << vsim->parentIndex() 
		<< std::endl;
    }

    /*
    std::cout <<  " simTrks   type, (momentum), vertIndex, genpartIndex"  << std::endl;
    int i=1;
    for(edm::SimTrackContainer::const_iterator t=simTrks->begin();
	t!=simTrks->end(); ++t){
      HepMC::GenParticle* gp=genEvt->particle( (*t).genpartIndex() );
	std::cout << i++ << ")" 
		  << (*t)
		  << " index="
		  << (*t).genpartIndex();
      if (gp) {
	HepMC::GenVertex *gv=gp->production_vertex();
	std::cout  <<  " genvertex =" << (*gv);
      }
      std::cout << std::endl;
    }
    */
  }

  // make a list of primary vertices:
  std::vector<simPrimaryVertex> simpv;
  // simvertices don't have enough information to decide, 
  // genVertices don't have the simulated coordinates 
  // go through simtracks to get the link between simulated and generated vertices
  // this might fail, if there are primary vertices which only produce particles that don't
  // make it into the simTracks. Is this possible?
  int idx=0;
  for(edm::SimTrackContainer::const_iterator t=simTrks->begin();
      t!=simTrks->end(); ++t){
    if ( !(t->noVertex()) && !(t->type()==-99) ){


      bool primary=false;     
      bool resonance=false;
      bool track=false;
      HepMC::GenParticle* gp=genEvt->particle( (*t).genpartIndex() );
      if (gp) {
	HepMC::GenVertex * gv=gp->production_vertex();
	if (gv->position().t()==0){
	  primary=true;
	}else if ( gp->mother() && isResonance(gp->mother())){
	  resonance=true;
	}
	if (gp->status()==1){
	  track=(HepPDT::theTable().getParticleData(gp->pdg_id())->charge() != 0);
	}
      }
      /* else
	 {
	// this simTrk does not have a GenParticle, it should have a parent in SimTrks then,
	// but pointers appear to be illegal occasionally => ignore them for now
	const SimVertex & simv=(*simVtcs)[t->vertIndex()];
	if ( (simv.parentIndex()>0) && (simv.parentIndex()<simTrks->size()) ){
	  const SimTrack &parent=(*simTrks)[simv.parentIndex()];
	  double cTau=HepPDT::theTable().getParticleData(parent.type())->cTau();
	  //std::cout << parent.type() << " ctau [mm]"  << cTau << std::endl;
	}
      }
      */

      const HepLorentzVector & v=(*simVtcs)[t->vertIndex()].position();
      if(primary or resonance){
	{
	  // check all primaries found so far to avoid multiple entries
	  bool newVertex=true;
	  for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	      v0!=simpv.end(); v0++){
	    if( (fabs(v0->x-v.x())<0.001) && (fabs(v0->y-v.y())<0.001) && (fabs(v0->z-v.z())<0.001) ){
	      if (track) v0->simTrackIndex.push_back(idx);
	      newVertex=false;
	    }
	  }
	  if(newVertex && !resonance){
	    simPrimaryVertex anotherVertex(v.x(),v.y(),v.z());
	    if (track) anotherVertex.simTrackIndex.push_back(idx);
	    simpv.push_back(anotherVertex);
	  }
	}// 
      }

    }// simtrack has vertex and valid type
    idx++;
  }//simTrack loop



  if(verbose_){
    for(std::vector<simPrimaryVertex>::const_iterator vsim=simpv.begin();
	vsim!=simpv.end(); vsim++){
      std::cout <<"primary " << vsim->x << " " << vsim->y << " " << vsim->z << " : ";
      for(unsigned int i=0; i<vsim->simTrackIndex.size(); i++){
	std::cout << (*simTrks)[vsim->simTrackIndex[i]].type() << " ";
      }
      std::cout << std::endl;
    }
  }





  // vertex matching and  efficiency accounting
  h["nsimvtx"]->Fill(simpv.size());
  h["nrecvtx"]->Fill(recVtxs->size());
  h["nsimtrk"]->Fill(simTrks->size());
  h["nrectrk"]->Fill(recTrks->size());
  for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
      vsim!=simpv.end(); vsim++){

    h["nbsimtksinvtx"]->Fill(vsim->simTrackIndex.size());
    h["xsim"]->Fill(vsim->x*simUnit_);
    h["ysim"]->Fill(vsim->y*simUnit_);
    h["zsim"]->Fill(vsim->z*simUnit_);

    // look for a matching reconstructed vertex
    vsim->recVtx=NULL;
    for(reco::VertexCollection::const_iterator vrec=recVtxs->begin(); 
	vrec!=recVtxs->end(); ++vrec){
      if ( matchVertex(*vsim,*vrec) ){
	if(    ((vsim->recVtx) && (fabs(vsim->recVtx->position().z()-vsim->z)>fabs(vrec->z()-vsim->z)))
	       || (!vsim->recVtx) )
	  {
	    vsim->recVtx=&(*vrec);
	  }
      }
    }

    if (vsim->recVtx){

      if(verbose_){std::cout <<"primary matched " << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}

      h["resx"]->Fill( vsim->recVtx->x()-vsim->x*simUnit_ );
      h["resy"]->Fill( vsim->recVtx->y()-vsim->y*simUnit_ );
      h["resz"]->Fill( vsim->recVtx->z()-vsim->z*simUnit_ );
      h["pullx"]->Fill( (vsim->recVtx->x()-vsim->x*simUnit_)/vsim->recVtx->xError() );
      h["pully"]->Fill( (vsim->recVtx->y()-vsim->y*simUnit_)/vsim->recVtx->yError() );
      h["pullz"]->Fill( (vsim->recVtx->z()-vsim->z*simUnit_)/vsim->recVtx->zError() );
      h["eff"]->Fill( 1.);
      
    }else{  // no rec vertex found for this simvertex

      if(verbose_){std::cout <<"primary not found " << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
      h["eff"]->Fill( 0.);

    }
  }
  // end of sim/rec matching 



  // test track links, use reconstructed vertices
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){

    try {
      for(reco::track_iterator t = v->tracks_begin(); 
	  t!=v->tracks_end(); t++) {
	// illegal charge
        if ( (**t).charge() < -1 || (**t).charge() > 1 ) {
	  h["tklinks"]->Fill(0.);
        }
        else {
	  h["tklinks"]->Fill(1.);
        }
      }
    }
    catch (...) {
      // exception thrown when trying to use linked track
      h["tklinks"]->Fill(0.);
    }

    h["nbvtx"]->Fill(recVtxs->size()*1.);
    h["nbtksinvtx"]->Fill(v->tracksSize());
    h["vtxchi2"]->Fill(v->chi2());
    h["vtxndf"]->Fill(v->ndof());
    h["xrec"]->Fill(v->position().x());
    h["yrec"]->Fill(v->position().y());
    h["zrec"]->Fill(v->position().z());

    bool problem = false;
    h["nans"]->Fill(1.,isnan(v->position().x())*1.);
    h["nans"]->Fill(2.,isnan(v->position().y())*1.);
    h["nans"]->Fill(3.,isnan(v->position().z())*1.);

    int index = 3;
    for (int i = 0; i != 3; i++) {
      for (int j = i; j != 3; j++) {
	index++;
	h["nans"]->Fill(index*1., isnan(v->covariance(i, j))*1.);
	if (isnan(v->covariance(i, j))) problem = true;
	// in addition, diagonal element must be positive
	if (j == i && v->covariance(i, j) < 0) {
	  h["nans"]->Fill(index*1., 1.);
	  problem = true;
	}
      }
    }

    //    if (problem) {
      // analyze track parameter covariance definiteness
      double data[25];
      try {
	int itk = 0;
	for(reco::track_iterator t = v->tracks_begin(); 
	    t!=v->tracks_end(); t++) {
	  std::cout << "Track " << itk++ << std::endl;
	  int i2 = 0;
	  for (int i = 0; i != 5; i++) {
	    for (int j = 0; j != 5; j++) {
	      data[i2] = (**t).covariance(i, j);
	      std::cout << data[i2] << " ";
	      i2++;
	    }
	    std::cout << std::endl;
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
      //}
  }
}
