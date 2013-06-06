#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

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
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

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
using namespace reco;
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
  TString tversion(edm::getReleaseVersion());
  tversion = tversion.Remove(0,1);
  tversion = tversion.Remove(tversion.Length()-1,tversion.Length());
  outputFile_  = std::string(tversion)+"_"+outputFile_;

  vtxSample_   = iConfig.getUntrackedParameter<std::vector< std::string > >("vtxSample");
  for(std::vector<std::string>::iterator isample = vtxSample_.begin(); isample!=vtxSample_.end(); ++isample) {
    if ( *isample == "offlinePrimaryVertices" ) suffixSample_.push_back("AVF");
    if ( *isample == "offlinePrimaryVerticesWithBS" ) suffixSample_.push_back("wBS");
  }
  if ( suffixSample_.size() == 0 ) throw cms::Exception("NoVertexSamples") << " no known vertex samples given";

  rootFile_ = new TFile(outputFile_.c_str(),"RECREATE");
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
   /*for(std::map<std::string,TH1*>::const_iterator hist=h.begin(); hist!=h.end(); hist++){
    delete hist->second;
    }*/
   
}



//
// member functions
//
void PrimaryVertexAnalyzer::beginJob(){
  std::cout << " PrimaryVertexAnalyzer::beginJob  conversion from sim units to rec units is " << simUnit_ << std::endl;


  rootFile_->cd();
  // release validation histograms used in DoCompare.C
  for (std::vector<std::string>::iterator isuffix= suffixSample_.begin(); isuffix!=suffixSample_.end(); isuffix++) {

    hdir[*isuffix] = rootFile_->mkdir(TString(*isuffix));
    hdir[*isuffix]->cd();
    h["nbvtx"+ *isuffix]        = new TH1F(TString("nbvtx"+ *isuffix),"Reconstructed Vertices in Event",20,-0.5,19.5);
    h["nbtksinvtx"+ *isuffix]   = new TH1F(TString("nbtksinvtx"+ *isuffix),"Reconstructed Tracks in Vertex",200,-0.5,199.5); 
    h["resx"+ *isuffix]         = new TH1F(TString("resx"+ *isuffix),"Residual X",100,-0.04,0.04);
    h["resy"+ *isuffix]         = new TH1F(TString("resy"+ *isuffix),"Residual Y",100,-0.04,0.04);
    h["resz"+ *isuffix]         = new TH1F(TString("resz"+ *isuffix),"Residual Z",100,-0.1,0.1);
    h["pullx"+ *isuffix]        = new TH1F(TString("pullx"+ *isuffix),"Pull X",100,-25.,25.);
    h["pully"+ *isuffix]        = new TH1F(TString("pully"+ *isuffix),"Pull Y",100,-25.,25.);
    h["pullz"+ *isuffix]        = new TH1F(TString("pullz"+ *isuffix),"Pull Z",100,-25.,25.);
    h["vtxchi2"+ *isuffix]      = new TH1F(TString("vtxchi2"+ *isuffix),"#chi^{2}",100,0.,100.);
    h["vtxndf"+ *isuffix]       = new TH1F(TString("vtxndf"+ *isuffix),"ndof",100,0.,100.);
    h["tklinks"+ *isuffix]      = new TH1F(TString("tklinks"+ *isuffix),"Usable track links",2,-0.5,1.5);
    h["nans"+ *isuffix]         = new TH1F(TString("nans"+ *isuffix),"Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
    // more histograms
    h["vtxprob"+ *isuffix]      = new TH1F(TString("vtxprob"+ *isuffix),"#chi^{2} probability",100,0.,1.);
    h["eff"+ *isuffix]          = new TH1F(TString("eff"+ *isuffix),"efficiency",2, -0.5, 1.5);
    h["efftag"+ *isuffix]       = new TH1F(TString("efftag"+ *isuffix),"efficiency tagged vertex",2, -0.5, 1.5);
    h["effvseta"+ *isuffix]     = new TProfile(TString("effvseta"+ *isuffix),"efficiency vs eta",20, -2.5, 2.5, 0, 1.);
    h["effvsptsq"+ *isuffix]    = new TProfile(TString("effvsptsq"+ *isuffix),"efficiency vs ptsq",20, 0., 10000., 0, 1.);
    h["effvsntrk"+ *isuffix]    = new TProfile(TString("effvsntrk"+ *isuffix),"efficiency vs # simtracks",200, 0., 200., 0, 1.);
    h["effvsnrectrk"+ *isuffix] = new TProfile(TString("effvsnrectrk"+ *isuffix),"efficiency vs # rectracks",200, 0., 200., 0, 1.);
    h["effvsz"+ *isuffix]       = new TProfile(TString("effvsz"+ *isuffix),"efficiency vs z",40, -20., 20., 0, 1.);
    h["nbsimtksinvtx"+ *isuffix]= new TH1F(TString("nbsimtksinvtx"+ *isuffix),"simulated tracks in vertex",100,-0.5,99.5); 
    h["xrec"+ *isuffix]         = new TH1F(TString("xrec"+ *isuffix),"reconstructed x",100,-0.1,0.1);
    h["yrec"+ *isuffix]         = new TH1F(TString("yrec"+ *isuffix),"reconstructed y",100,-0.1,0.1);
    h["zrec"+ *isuffix]         = new TH1F(TString("zrec"+ *isuffix),"reconstructed z",100,-20.,20.);
    h["xsim"+ *isuffix]         = new TH1F(TString("xsim"+ *isuffix),"simulated x",100,-0.1,0.1);
    h["ysim"+ *isuffix]         = new TH1F(TString("ysim"+ *isuffix),"simulated y",100,-0.1,0.1);
    h["zsim"+ *isuffix]         = new TH1F(TString("zsim"+ *isuffix),"simulated z",100,-20.,20.);
    h["nrecvtx"+ *isuffix]      = new TH1F(TString("nrecvtx"+ *isuffix),"# of reconstructed vertices", 50, -0.5, 49.5);
    h["nsimvtx"+ *isuffix]      = new TH1F(TString("nsimvtx"+ *isuffix),"# of simulated vertices", 50, -0.5, 49.5);
    h["nrectrk"+ *isuffix]      = new TH1F(TString("nrectrk"+ *isuffix),"# of reconstructed tracks", 200, -0.5, 199.5);
    h["nsimtrk"+ *isuffix]      = new TH1F(TString("nsimtrk"+ *isuffix),"# of simulated tracks", 200, -0.5, 199.5);
    h["xrectag"+ *isuffix]      = new TH1F(TString("xrectag"+ *isuffix),"reconstructed x, signal vtx",100,-0.1,0.1);
    h["yrectag"+ *isuffix]      = new TH1F(TString("yrectag"+ *isuffix),"reconstructed y, signal vtx",100,-0.1,0.1);
    h["zrectag"+ *isuffix]      = new TH1F(TString("zrectag"+ *isuffix),"reconstructed z, signal vtx",100,-20.,20.);
    h["rapidity"+ *isuffix] = new TH1F(TString("rapidity"+ *isuffix),"rapidity ",100,-10., 10.);
    h["pt"+ *isuffix] = new TH1F(TString("pt"+ *isuffix),"pt ",100,0., 20.);
    h["nrectrk0vtx"+ *isuffix] = new TH1F(TString("nrectrk0vtx"+ *isuffix),"# rec tracks no vertex ",200,0., 200.);
    h["zdistancetag"+ *isuffix] = new TH1F(TString("zdistancetag"+ *isuffix),"z-distance between tagged and generated",100, -0.1, 0.1);
    h["puritytag"+ *isuffix]    = new TH1F(TString("puritytag"+ *isuffix),"purity of primary vertex tags",2, -0.5, 1.5);
  }

}


void PrimaryVertexAnalyzer::endJob() {
  rootFile_->cd();
  // save all histograms created in beginJob()
  for (std::map<std::string, TDirectory*>::const_iterator idir=hdir.begin(); idir!=hdir.end(); idir++){
    idir->second->cd();
    for(std::map<std::string,TH1*>::const_iterator hist=h.begin(); hist!=h.end(); hist++){
      if (TString(hist->first).Contains(idir->first)) hist->second->Write();
    }
  }
}


// helper functions
bool PrimaryVertexAnalyzer::matchVertex(const simPrimaryVertex  &vsim, 
				       const reco::Vertex       &vrec){
  return (fabs(vsim.z*simUnit_-vrec.z())<0.0500); // =500um
}

bool PrimaryVertexAnalyzer::isResonance(const HepMC::GenParticle * p){
  double ctau=(pdt->particle( abs(p->pdg_id()) ))->lifetime();
  if(verbose_) std::cout << "isResonance   " << p->pdg_id() << " " << ctau << std::endl;
  return  ctau >0 && ctau <1e-6;
}

bool PrimaryVertexAnalyzer::isFinalstateParticle(const HepMC::GenParticle * p){
  return ( !p->end_vertex() && p->status()==1 );
}

bool PrimaryVertexAnalyzer::isCharged(const HepMC::GenParticle * p){
  const ParticleData * part = pdt->particle( p->pdg_id() );
  if (part){
    return part->charge()!=0;
  }else{
    // the new/improved particle table doesn't know anti-particles
    return  pdt->particle( -p->pdg_id() )!=0;
  }
}

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
				      const Handle<HepMCProduct> evtMC, std::string suffix="")
{
  std::vector<PrimaryVertexAnalyzer::simPrimaryVertex> simpv;
  const HepMC::GenEvent* evt=evtMC->GetEvent();
  if (evt) {
    if(verbose_) std::cout << "process id " <<evt->signal_process_id()<<std::endl;
    if(verbose_) std::cout <<"signal process vertex "<< ( evt->signal_process_vertex() ?
					     evt->signal_process_vertex()->barcode() : 0 )   <<std::endl;
    if(verbose_) std::cout <<"number of vertices " << evt->vertices_size() << std::endl;


    int idx=0;
    for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
	vitr != evt->vertices_end(); ++vitr ) 
      { // loop for vertex ...
	HepMC::FourVector pos = (*vitr)->position();
	//HepLorentzVector pos = (*vitr)->position();

	bool hasMotherVertex=false;
	if(verbose_) std::cout << "mothers of vertex[" << ++idx << "]: " << std::endl;
	for ( HepMC::GenVertex::particle_iterator
	      mother  = (*vitr)->particles_begin(HepMC::parents);
	      mother != (*vitr)->particles_end(HepMC::parents);
              ++mother ) {
	  HepMC::GenVertex * mv=(*mother)->production_vertex();
	  if (mv) {
	    hasMotherVertex=true;
	    if(!verbose_) break; //if verbose_, print all particles of gen vertices
	  }
	  if(verbose_)std::cout << "\t";
	  if(verbose_)(*mother)->print();
	}

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
	  if(verbose_)std::cout << "this is a new vertex " << sv.x << " " << sv.y << " " << sv.z << std::endl;
	  simpv.push_back(sv);
	  vp=&simpv.back();
	}else{
	  if(verbose_)std::cout << "this is not new vertex " << sv.x << " " << sv.y << " " << sv.z << std::endl;
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
	      HepMC::FourVector m=(*daughter)->momentum();
	      // the next four lines used to be "vp->ptot+=m;" in the days of CLHEP::HepLorentzVector
	      // but adding FourVectors seems not to be foreseen
	      vp->ptot.setPx(vp->ptot.px()+m.px());
	      vp->ptot.setPy(vp->ptot.py()+m.py());
	      vp->ptot.setPz(vp->ptot.pz()+m.pz());
	      vp->ptot.setE(vp->ptot.e()+m.e());
	      vp->ptsq+=(m.perp())*(m.perp());
	      if ( (m.perp()>0.8) && (fabs(m.pseudoRapidity())<2.5) && isCharged( *daughter ) ){
		vp->nGenTrk++;
	      }
	      
	      h["rapidity"+suffix]->Fill(m.pseudoRapidity());
	      h["pt"+suffix]->Fill(m.perp());
	    }
	  }
	}
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
       HepMC::GenParticle* gp=evtMC->GetEvent()->barcode_to_particle( (*t).genpartIndex() );
       if (gp) {
	 HepMC::GenVertex * gv=gp->production_vertex();
	 if (gv) {
	   for ( HepMC::GenVertex::particle_iterator 
                 daughter =  gv->particles_begin(HepMC::descendants);
		 daughter != gv->particles_end(HepMC::descendants);
		 ++daughter ) {
	     if (isFinalstateParticle(*daughter)){
	       ptsq+=(*daughter)->momentum().perp()*(*daughter)->momentum().perp();
	     }
	   }
	   primary =  ( gv->position().t()==0);
	   //resonance= ( gp->mother() && isResonance(gp->mother()));  // in CLHEP/HepMC days
	   // no more mother pointer in the improved HepMC GenParticle
	   resonance= ( isResonance(*(gp->production_vertex()->particles_in_const_begin())));
	   if (gp->status()==1){
	     //track=((pdt->particle(gp->pdg_id()))->charge() != 0);
	     track=not isCharged(gp);
	   }
	 }
       }

       const HepMC::FourVector & v=(*simVtxs)[t->vertIndex()].position();
       //const HepLorentzVector & v=(*simVtxs)[t->vertIndex()].position();
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
  
  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);

  Handle<SimVertexContainer> simVtxs;
  iEvent.getByLabel( simG4_, simVtxs);
  
  Handle<SimTrackContainer> simTrks;
  iEvent.getByLabel( simG4_, simTrks);

  for (int ivtxSample=0; ivtxSample!= (int)vtxSample_.size(); ++ivtxSample) {
    std::string isuffix = suffixSample_[ivtxSample];

    Handle<reco::VertexCollection> recVtxs;
    iEvent.getByLabel(vtxSample_[ivtxSample], recVtxs);

    try{
      iSetup.getData(pdt);
    }catch(const Exception&){
      std::cout << "Some problem occurred with the particle data table. This may not work !." <<std::endl;
    }

    bool MC=false;
    Handle<HepMCProduct> evtMC;
    iEvent.getByLabel("generator",evtMC);
    if (!evtMC.isValid()) {
      MC=false;
      if(verbose_){
	std::cout << "no HepMCProduct found"<< std::endl;
      }
    } else {
      if(verbose_){
	std::cout << "generator HepMCProduct found"<< std::endl;
      }
      MC=true;
    }
    
    
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
      simpv=getSimPVs(evtMC,isuffix);
      
      
      // vertex matching and efficiency bookkeeping
      h["nsimvtx"+isuffix]->Fill(simpv.size());
      int nsimtrk=0;
      for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
	  vsim!=simpv.end(); vsim++){
	
	hdir[isuffix]->cd();
	
	h["nbsimtksinvtx"+isuffix]->Fill(vsim->nGenTrk);
	h["xsim"+isuffix]->Fill(vsim->x*simUnit_);
	h["ysim"+isuffix]->Fill(vsim->y*simUnit_);
	h["zsim"+isuffix]->Fill(vsim->z*simUnit_);
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
	h["nsimtrk"+isuffix]->Fill(float(nsimtrk));
	
	
	// histogram properties of matched vertices
	if (vsim->recVtx){
	  
	  if(verbose_){std::cout <<"primary matched " << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
	  
	  h["resx"+isuffix]->Fill( vsim->recVtx->x()-vsim->x*simUnit_ );
	  h["resy"+isuffix]->Fill( vsim->recVtx->y()-vsim->y*simUnit_ );
	  h["resz"+isuffix]->Fill( vsim->recVtx->z()-vsim->z*simUnit_ );
	  h["pullx"+isuffix]->Fill( (vsim->recVtx->x()-vsim->x*simUnit_)/vsim->recVtx->xError() );
	  h["pully"+isuffix]->Fill( (vsim->recVtx->y()-vsim->y*simUnit_)/vsim->recVtx->yError() );
	  h["pullz"+isuffix]->Fill( (vsim->recVtx->z()-vsim->z*simUnit_)/vsim->recVtx->zError() );
	  h["eff"+isuffix]->Fill( 1.);
	  if(simpv.size()==1){
	    if (vsim->recVtx==&(*recVtxs->begin())){
	      h["efftag"+isuffix]->Fill( 1.); 
	    }else{
	      h["efftag"+isuffix]->Fill( 0.); 
	    }
	  }
	  h["effvseta"+isuffix]->Fill(vsim->ptot.pseudoRapidity(),1.);
	  h["effvsptsq"+isuffix]->Fill(vsim->ptsq,1.);
	  h["effvsntrk"+isuffix]->Fill(vsim->nGenTrk,1.);
	  h["effvsnrectrk"+isuffix]->Fill(recTrks->size(),1.);
	  h["effvsz"+isuffix]->Fill(vsim->z*simUnit_,1.);
	  
	}else{  // no rec vertex found for this simvertex
	  
	  if(verbose_){std::cout <<"primary not found " << vsim->x << " " << vsim->y << " " << vsim->z << " nGenTrk=" << vsim->nGenTrk << std::endl;}
	  
	  h["eff"+isuffix]->Fill( 0.);
	  if(simpv.size()==1){ h["efftag"+isuffix]->Fill( 0.); }
	  h["effvseta"+isuffix]->Fill(vsim->ptot.pseudoRapidity(),0.);
	  h["effvsptsq"+isuffix]->Fill(vsim->ptsq,0.);
	  h["effvsntrk"+isuffix]->Fill(float(vsim->nGenTrk),0.);
	  h["effvsnrectrk"+isuffix]->Fill(recTrks->size(),0.);
	  h["effvsz"+isuffix]->Fill(vsim->z*simUnit_,0.);
	} // no recvertex for this simvertex
      }//found MC event
      // end of sim/rec matching 
      
      
      // purity of event vertex tags
      if (recVtxs->size()>0 && simpv.size()>0){
	Double_t dz=(*recVtxs->begin()).z() - (*simpv.begin()).z*simUnit_;
	h["zdistancetag"+isuffix]->Fill(dz);
	if( fabs(dz)<0.0500){
	  h["puritytag"+isuffix]->Fill(1.);
	}else{
	  // bad tag: the true primary was more than 500 um away from the tagged primary
	  h["puritytag"+isuffix]->Fill(0.);
	}
      }
      
      
    }// MC event
    
    // test track links, use reconstructed vertices
    
    h["nrecvtx"+isuffix]->Fill(recVtxs->size());
    h["nrectrk"+isuffix]->Fill(recTrks->size());
    h["nbvtx"+isuffix]->Fill(recVtxs->size()*1.);
    if((recVtxs->size()==0)||recVtxs->begin()->isFake()) {h["nrectrk0vtx"+isuffix]->Fill(recTrks->size());}
    
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){

      if(v->isFake()) continue;
      
      try {
	for(reco::Vertex::trackRef_iterator t = v->tracks_begin(); 
	    t!=v->tracks_end(); t++) {
	  // illegal charge
	  if ( (**t).charge() < -1 || (**t).charge() > 1 ) {
	    h["tklinks"+isuffix]->Fill(0.);
	  }
	  else {
	    h["tklinks"+isuffix]->Fill(1.);
	  }
	}
      }
      catch (...) {
	// exception thrown when trying to use linked track
	h["tklinks"+isuffix]->Fill(0.);
      }
      
      h["nbtksinvtx"+isuffix]->Fill(v->tracksSize());
      h["vtxchi2"+isuffix]->Fill(v->chi2());
      h["vtxndf"+isuffix]->Fill(v->ndof());
      h["vtxprob"+isuffix]->Fill(ChiSquaredProbability(v->chi2() ,v->ndof()));
      h["xrec"+isuffix]->Fill(v->position().x());
      h["yrec"+isuffix]->Fill(v->position().y());
      h["zrec"+isuffix]->Fill(v->position().z());
      if (v==recVtxs->begin()){
	h["xrectag"+isuffix]->Fill(v->position().x());
	h["yrectag"+isuffix]->Fill(v->position().y());
	h["zrectag"+isuffix]->Fill(v->position().z());
      }
      
      bool problem = false;
      h["nans"+isuffix]->Fill(1.,isnan(v->position().x())*1.);
      h["nans"+isuffix]->Fill(2.,isnan(v->position().y())*1.);
      h["nans"+isuffix]->Fill(3.,isnan(v->position().z())*1.);
      
      int index = 3;
      for (int i = 0; i != 3; i++) {
	for (int j = i; j != 3; j++) {
	  index++;
	  h["nans"+isuffix]->Fill(index*1., isnan(v->covariance(i, j))*1.);
	  if (isnan(v->covariance(i, j))) problem = true;
	  // in addition, diagonal element must be positive
	  if (j == i && v->covariance(i, j) < 0) {
	    h["nans"+isuffix]->Fill(index*1., 1.);
	    problem = true;
	  }
	}
      }
      
      if (problem) {
	// analyze track parameter covariance definiteness
	double data[25];
	try {
	  int itk = 0;
	  for(reco::Vertex::trackRef_iterator t = v->tracks_begin(); 
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
		//gsl_vector_view evec_i 
		//  = gsl_matrix_column (evec, i);
		
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
  } // for vertex loop
}
