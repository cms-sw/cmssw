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

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

// Root
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TProfile.h>
 
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexAnalyzer::PrimaryVertexAnalyzer(const ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  simG4_=iConfig.getParameter<edm::InputTag>( "simG4" );
  recoTrackProducer_= iConfig.getUntrackedParameter<std::string>("recoTrackProducer");
  // open output file to store histograms}
  outputFile_  = iConfig.getUntrackedParameter<std::string>("outputFile");
  vtxSample_   = iConfig.getUntrackedParameter<std::string>("vtxSample");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  verbose_= iConfig.getUntrackedParameter<bool>("verbose", false);
  simUnit_= 1.0;  // starting with CMSSW_1_2_x ??
  if ( (edm::getReleaseVersion()).find("CMSSW_1_1_",0)!=std::string::npos){
    simUnit_=0.1;  // for use in  CMSSW_1_1_1 tutorial
  }
}


PrimaryVertexAnalyzer::~PrimaryVertexAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete rootFile_;
  /*
  for(std::map<std::string,TH1*>::const_iterator hist=h.begin(); hist!=h.end(); hist++){
    delete hist->second;
  }
  */
}



//
// member functions
//
void PrimaryVertexAnalyzer::beginJob(edm::EventSetup const& iSetup){
  std::cout << " PrimaryVertexAnalyzer::beginJob  conversion from sim units to rec units is " << simUnit_ << std::endl;


  rootFile_->cd();
  // release validation histograms used in DoCompare.C
  h["nbvtx"]        = new TH1F("nbvtx","nb rec vertices in event",20,-0.5,19.5);
  h["nbtksinvtx"]   = new TH1F("nbtksinvtx","reconstructed tracks in vertex",40,-0.5,39.5); 
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
  h["vtxprob"]      = new TH1F("vtxprob","chisquared probability",100,0.,1.);
  h["eff"]          = new TH1F("eff","efficiency",2, -0.5, 1.5);
  h["efftag"]       = new TH1F("efftag","efficiency tagged vertex",2, -0.5, 1.5);
  h["effvseta"]     = new TProfile("effvseta","efficiency vs eta",20, -2.5, 2.5, 0, 1.);
  h["effvsptsq"]    = new TProfile("effvsptsq","efficiency vs ptsq",20, 0., 10000., 0, 1.);
  h["effvsntrk"]    = new TProfile("effvsntrk","efficiency vs # tracks",50, 0., 50., 0, 1.);
  h["effvsnrectrk"] = new TProfile("effvsnrectrk","efficiency vs # rectracks",50, 0., 50., 0, 1.);
  h["effvsz"]       = new TProfile("effvsz","efficiency vs z",40, -20., 20., 0, 1.);
  h["nbsimtksinvtx"]= new TH1F("nbsimtksinvtx","simulated tracks in vertex",100,-0.5,99.5); 
  h["xrec"]         = new TH1F("xrec","reconstructed x",100,-0.01,0.01);
  h["yrec"]         = new TH1F("yrec","reconstructed y",100,-0.01,0.01);
  h["zrec"]         = new TH1F("zrec","reconstructed z",100,-20.,20.);
  h["xsim"]         = new TH1F("xsim","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  h["ysim"]         = new TH1F("ysim","simulated y",100,-0.01,0.01);
  h["zsim"]         = new TH1F("zsim","simulated z",100,-20.,20.);
  h["nrecvtx"]      = new TH1F("nrecvtx","# of reconstructed vertices", 50, -0.5, 49.5);
  h["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  h["nrectrk"]      = new TH1F("nrectrk","# of reconstructed tracks", 50, -0.5, 49.5);
  h["nsimtrk"]      = new TH1F("nsimtrk","# of simulated tracks", 50, -0.5, 49.5);
  h["xrectag"]      = new TH1F("xrectag","reconstructed x, signal vtx",100,-0.01,0.01);
  h["yrectag"]      = new TH1F("yrectag","reconstructed y, signal vtx",100,-0.01,0.01);
  h["zrectag"]      = new TH1F("zrectag","reconstructed z, signal vtx",100,-20.,20.);
  h["rapidity"] = new TH1F("rapidity","rapidity ",100,-10., 10.);
  h["pt"] = new TH1F("pt","pt ",100,0., 20.);
  h["nrectrk0vtx"] = new TH1F("nrectrk0vtx","# rec tracks no vertex ",50,0., 50.);
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

bool PrimaryVertexAnalyzer::isFinalstateParticle(const HepMC::GenParticle * p){
  return ( !p->end_vertex() && p->status()==1 );
}
/*
bool PrimaryVertexAnalyzer::isPrimaryParticle(const HepMC::GenParticle * p){
  HepMC::GenVertex * gv=gp->production_vertex();
  if (gv){
    if (position().t()==0){
      return true;
    }else{
      return gp->mother() && isResonane(gp->mother()) && isPrimaryParticle(gp->mother());
    }
  }else{
    std::cout << "particle has no production vertex " << p->pdg_id() << std::endl;
    return false; // don't know much about it
  }
}
*/

void PrimaryVertexAnalyzer::printRecVtxs(const Handle<reco::VertexCollection> recVtxs){
    int ivtx=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      std::cout << "Recvtx "<< std::setw(3) << std::setfill(' ')<<ivtx++
		<< "#trk " << std::setw(3) << v->tracksSize() 
		<< " chi2 " << std::setw(4) << v->chi2() 
		<< " ndof " << std::setw(3) << v->ndof() << std::endl 
		<< " x "  << std::setw(8) <<std::fixed << std::setprecision(4) << v->x() 
		<< " dx " << std::setw(8) << v->xError()<< std::endl
		<< " y "  << std::setw(8) << v->y() 
		<< " dy " << std::setw(8) << v->yError()<< std::endl
		<< " z "  << std::setw(8) << v->z() 
		<< " dz " << std::setw(8) << v->zError()
		<< std::endl;
    }
}


void PrimaryVertexAnalyzer::printSimVtxs(const Handle<SimVertexContainer> simVtxs){
    int i=0;
    for(SimVertexContainer::const_iterator vsim=simVtxs->begin();
	vsim!=simVtxs->end(); ++vsim){
      std::cout << i++ << ")" << std::scientific
                << " evtid=" << vsim->eventId().event() 
		<< " sim x=" << vsim->position().x()*simUnit_
		<< " sim y=" << vsim->position().y()*simUnit_
		<< " sim z=" << vsim->position().z()*simUnit_
		<< " sim t=" << vsim->position().t()
		<< " parent=" << vsim->parentIndex() 
		<< std::endl;
    }
}


void PrimaryVertexAnalyzer::printSimTrks(const Handle<SimTrackContainer> simTrks){
  std::cout <<  " simTrks   type, (momentum), vertIndex, genpartIndex"  << std::endl;
  int i=1;
  for(SimTrackContainer::const_iterator t=simTrks->begin();
      t!=simTrks->end(); ++t){
    //HepMC::GenParticle* gp=evtMC->GetEvent()->particle( (*t).genpartIndex() );
    std::cout << i++ << ")" 
	      << (*t)
	      << " index="
	      << (*t).genpartIndex();
    //if (gp) {
    //  HepMC::GenVertex *gv=gp->production_vertex();
    //  std::cout  <<  " genvertex =" << (*gv);
    //}
    std::cout << std::endl;
  }
}


std::vector<PrimaryVertexAnalyzer::simPrimaryVertex> PrimaryVertexAnalyzer::getSimPVs(
				      const Handle<HepMCProduct> evtMC)
{
  std::vector<PrimaryVertexAnalyzer::simPrimaryVertex> simpv;
  const HepMC::GenEvent* evt=evtMC->GetEvent();
  if (evt) {
    std::cout << "process id " <<evt->signal_process_id()<<std::endl;
    std::cout <<"signal process vertex "<< ( evt->signal_process_vertex() ?
					     evt->signal_process_vertex()->barcode() : 0 )   <<std::endl;
    std::cout <<"number of vertices " << evt->vertices_size() << std::endl;
    //std::cout <<"isVtxGenApplied   " << evtMC->isVtxGenApplied() << std::endl; 


    int idx=0;
    for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
	vitr != evt->vertices_end(); ++vitr ) 
      { // loop for vertex ...
	//std::cout << "looking at vertex " << idx << std::endl;
	//std::cout << "has parents  " <<(*vitr)->hasParents() << " n= " <<  (*vitr)->numParents()<< std::endl;
	//std::cout << "has children  " << (*vitr)->hasChildren() <<  " n= " <<  (*vitr)->numChildren()<<  std::endl;
	
	HepLorentzVector pos = (*vitr)->position();
	if (pos.t()>0) { continue;}

	bool hasMotherVertex=false;
	//std::cout << "mothers" << std::endl;
	for ( HepMC::GenVertex::particle_iterator
	      mother  = (*vitr)->particles_begin(HepMC::parents);
	      mother != (*vitr)->particles_end(HepMC::parents);
              ++mother ) {
	  HepMC::GenVertex * mv=(*mother)->production_vertex();
	  if (mv) {hasMotherVertex=true;}
	  //std::cout << "\t";
	  //(*mother)->print();
	}
	/*
	std::cout << "daughters" << std::endl;
	for ( HepMC::GenVertex::particle_iterator
	      daughter  = (*vitr)->particles_begin(HepMC::children);
	      daughter != (*vitr)->particles_end(HepMC::children);
              ++daughter ) {
	  (*daughter)->print();
	}
	std::cout << " hasMotherVertex " << hasMotherVertex <<std::endl;
	*/

	if(hasMotherVertex) {continue;}

	// could be a new vertex, check  all primaries found so far to avoid multiple entries
        const double mm=0.1;
	simPrimaryVertex sv(pos.x()*mm,pos.y()*mm,pos.z()*mm);
	simPrimaryVertex *vp=NULL;  // will become non-NULL if a vertex is found and then point to it
	for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	    v0!=simpv.end(); v0++){
	  if( (fabs(sv.x-v0->x)<1e-5) && (fabs(sv.y-v0->y)<1e-5) && (fabs(sv.z-v0->z)<1e-5)){
	    vp=&(*v0);
	    break;
	  }
	}

	if(!vp){
	  // this is a new vertex
	  //std::cout << "this is a new vertex" << sv.x << " " << sv.y << " " << sv.z <<std::endl;
	  simpv.push_back(sv);
	  vp=&simpv.back();
	}else{
	  //std::cout << "this is not new vertex" << std::endl;
	}
	vp->genVertex.push_back((*vitr)->barcode());
	// collect final state descendants
	for ( HepMC::GenVertex::particle_iterator
	      daughter  = (*vitr)->particles_begin(HepMC::descendants);
	      daughter != (*vitr)->particles_end(HepMC::descendants);
              ++daughter ) {
	  if (isFinalstateParticle(*daughter)){ 
	    if ( find(vp->finalstateParticles.begin(), vp->finalstateParticles.end(),(*daughter)->barcode())
		 == vp->finalstateParticles.end()){
	      vp->finalstateParticles.push_back((*daughter)->barcode());
	      HepLorentzVector m=(*daughter)->momentum();
	      vp->ptot+=m;
	      vp->ptsq+=(m.perp())*(m.perp());
	      if ( (m.perp()>0.8) && fabs(m.rapidity()<2.5) 
		   && (HepPDT::theTable().getParticleData((*daughter)->pdg_id())->charge() != 0)){
		vp->nGenTrk++;
	      }
	      h["rapidity"]->Fill(m.rapidity());
	      h["pt"]->Fill(m.perp());
	    }
	    //std::cout << (*daughter)->barcode() << "\t";
	  }
	}
	idx++;
      }
  }
  return simpv;
}



std::vector<PrimaryVertexAnalyzer::simPrimaryVertex> PrimaryVertexAnalyzer::getSimPVs(
				      const Handle<HepMCProduct> evtMC, 
				      const Handle<SimVertexContainer> simVtxs, 
				      const Handle<SimTrackContainer> simTrks)
{
   // simvertices don't have enough information to decide, 
   // genVertices don't have the simulated coordinates  ( with VtxSmeared they might)
   // go through simtracks to get the link between simulated and generated vertices
   std::vector<PrimaryVertexAnalyzer::simPrimaryVertex> simpv;
   int idx=0;
   for(SimTrackContainer::const_iterator t=simTrks->begin();
       t!=simTrks->end(); ++t){
     if ( !(t->noVertex()) && !(t->type()==-99) ){
       double ptsq=0;
       bool primary=false;   // something coming directly from the primary vertex
       bool resonance=false; // resonance
       bool track=false;     // undecayed, charged particle
       HepMC::GenParticle* gp=evtMC->GetEvent()->particle( (*t).genpartIndex() );
       if (gp) {
	 HepMC::GenVertex * gv=gp->production_vertex();
	 if (gv) {
	   for ( HepMC::GenVertex::particle_iterator 
                 daughter =  gv->particles_begin(HepMC::descendants);
		 daughter != gv->particles_end(HepMC::descendants);
		 ++daughter ) {
	     //(*daughter)->print();
	     if (isFinalstateParticle(*daughter)){
	       ptsq+=(*daughter)->momentum().perp()*(*daughter)->momentum().perp();
	     }
	   }
	   primary =  ( gv->position().t()==0);
	   resonance= ( gp->mother() && isResonance(gp->mother()));
	   if (gp->status()==1){
	     track=(HepPDT::theTable().getParticleData(gp->pdg_id())->charge() != 0);
	   }
	 }
       }

       const HepLorentzVector & v=(*simVtxs)[t->vertIndex()].position();
       if(primary or resonance){
	 {
	   // check all primaries found so far to avoid multiple entries
	   bool newVertex=true;
	   for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	       v0!=simpv.end(); v0++){
	     if( (fabs(v0->x-v.x())<0.001) && (fabs(v0->y-v.y())<0.001) && (fabs(v0->z-v.z())<0.001) ){
	       if (track) {
		 v0->simTrackIndex.push_back(idx);
		 if (ptsq>(*v0).ptsq){(*v0).ptsq=ptsq;}
	       }
	       newVertex=false;
	     }
	   }
	   if(newVertex && !resonance){
	     simPrimaryVertex anotherVertex(v.x(),v.y(),v.z());
	     if (track) anotherVertex.simTrackIndex.push_back(idx);
	     anotherVertex.ptsq=ptsq;
	     simpv.push_back(anotherVertex);
	   }
	 }// 
       }
       
     }// simtrack has vertex and valid type
     idx++;
   }//simTrack loop
   return simpv;
}




// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel(vtxSample_, recVtxs);
  
  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);

  Handle<SimVertexContainer> simVtxs;
  iEvent.getByLabel( simG4_, simVtxs);
  
  Handle<SimTrackContainer> simTrks;
  iEvent.getByLabel( simG4_, simTrks);

  bool MC=false;
  Handle<HepMCProduct> evtMC;
  try{
    iEvent.getByLabel("VtxSmeared",evtMC);
    MC=true;
    if(verbose_){
      std::cout << "VtxSmeared HepMCProduct found"<< std::endl;
    }
    //if(evtMC->GetEvent()){ evtMC->GetEvent()->print();
  }catch(const Exception&){
    // VtxSmeared not found, try source
    try{
      iEvent.getByLabel("source",evtMC);
      if(verbose_){
	std::cout << "source HepMCProduct found"<< std::endl;
      }
      MC=true;
    }catch(const Exception&) {
      MC=false;
      if(verbose_){
	std::cout << "no HepMCProduct found"<< std::endl;
      }
    }
  }

  /*
  if(evtMC->GetEvent()->signal_process_vertex()){
    HepLorentzVector vsig=evtMC->GetEvent()->signal_process_vertex()->position();
    std::cout <<" signal vertex " << vsig.x() << " " << vsig.z() << std::endl;
  }else{
    std::cout <<" no signal vertex"  << std::endl;
  }
  */

  if(verbose_){
    //evtMC->GetEvent()->print();
    printRecVtxs(recVtxs);
    //printSimVtxs(simVtxs);
    //printSimTrks(simTrks);
  }

  if(MC){

   // make a list of primary vertices:
   std::vector<simPrimaryVertex> simpv;
   //simpv=getSimPVs(evtMC, simVtxs, simTrks);
   simpv=getSimPVs(evtMC);

   /*
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
   */

   // vertex matching and efficiency bookkeeping
   h["nsimvtx"]->Fill(simpv.size());
   int nsimtrk=0;
   for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
       vsim!=simpv.end(); vsim++){
     
     h["nbsimtksinvtx"]->Fill(vsim->nGenTrk);
     h["xsim"]->Fill(vsim->x*simUnit_);
     h["ysim"]->Fill(vsim->y*simUnit_);
     h["zsim"]->Fill(vsim->z*simUnit_);
     nsimtrk+=vsim->nGenTrk;
     // look for a matching reconstructed vertex
     vsim->recVtx=NULL;
     for(reco::VertexCollection::const_iterator vrec=recVtxs->begin(); 
	 vrec!=recVtxs->end(); ++vrec){
       if ( matchVertex(*vsim,*vrec) ){
	 // if the matching critera are fulfilled, accept the rec-vertex that is closest in z
	 if(    ((vsim->recVtx) && (fabs(vsim->recVtx->position().z()-vsim->z)>fabs(vrec->z()-vsim->z)))
		|| (!vsim->recVtx) )
	   {
	     vsim->recVtx=&(*vrec);
	   }
       }
     }
     h["nsimtrk"]->Fill(float(nsimtrk));
     
       
     // histogram properties of matched vertices
     if (vsim->recVtx){
       
       if(verbose_){std::cout <<"primary matched " << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
       
       h["resx"]->Fill( vsim->recVtx->x()-vsim->x*simUnit_ );
       h["resy"]->Fill( vsim->recVtx->y()-vsim->y*simUnit_ );
       h["resz"]->Fill( vsim->recVtx->z()-vsim->z*simUnit_ );
       h["pullx"]->Fill( (vsim->recVtx->x()-vsim->x*simUnit_)/vsim->recVtx->xError() );
       h["pully"]->Fill( (vsim->recVtx->y()-vsim->y*simUnit_)/vsim->recVtx->yError() );
       h["pullz"]->Fill( (vsim->recVtx->z()-vsim->z*simUnit_)/vsim->recVtx->zError() );
       h["eff"]->Fill( 1.);
       if((simpv.size()==1)&&(vsim->recVtx==&(*recVtxs->begin()))){ h["efftag"]->Fill( 1.); }
       h["effvseta"]->Fill(vsim->ptot.rapidity(),1.);
       h["effvsptsq"]->Fill(vsim->ptsq,1.);
       h["effvsntrk"]->Fill(vsim->nGenTrk,1.);
       h["effvsnrectrk"]->Fill(recTrks->size(),1.);
       h["effvsz"]->Fill(vsim->z*simUnit_,1.);
       
     }else{  // no rec vertex found for this simvertex
       
       if(verbose_){std::cout <<"primary not found " << vsim->x << " " << vsim->y << " " << vsim->z << " nGenTrk=" << vsim->nGenTrk << std::endl;}

       h["eff"]->Fill( 0.);
       if((simpv.size()==1)&&(vsim->recVtx==&(*recVtxs->begin()))){ h["efftag"]->Fill( 0.); }
       h["effvseta"]->Fill(vsim->ptot.rapidity(),0.);
       h["effvsptsq"]->Fill(vsim->ptsq,0.);
       h["effvsntrk"]->Fill(float(vsim->nGenTrk),0.);
       h["effvsnrectrk"]->Fill(recTrks->size(),0.);
       h["effvsz"]->Fill(vsim->z*simUnit_,0.);
     }
   }
  }//found MC event
  // end of sim/rec matching 



  // test track links, use reconstructed vertices

  h["nrecvtx"]->Fill(recVtxs->size());
  h["nrectrk"]->Fill(recTrks->size());
  if(recVtxs->size()==0) {h["nrectrk0vtx"]->Fill(recTrks->size());}

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
    h["vtxprob"]->Fill(ChiSquaredProbability(v->chi2() ,v->ndof()));
    h["xrec"]->Fill(v->position().x());
    h["yrec"]->Fill(v->position().y());
    h["zrec"]->Fill(v->position().z());
    if (v==recVtxs->begin()){
      h["xrectag"]->Fill(v->position().x());
      h["yrectag"]->Fill(v->position().y());
      h["zrectag"]->Fill(v->position().z());
    }

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

    if (problem) {
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
	      std::cout << std:: scientific << data[i2] << " ";
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
    }// if (problem)
  }
}
