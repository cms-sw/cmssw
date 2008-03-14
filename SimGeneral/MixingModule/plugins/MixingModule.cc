// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "MixingModule.h"

const int  edm::MixingModule::lowTrackTof = -36; 
const int  edm::MixingModule::highTrackTof = 36; 
const int  edm::MixingModule::limHighLowTof = 36; 


using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps) : BMixingModule(ps),
							    label_(ps.getParameter<std::string>("Label"))

  {
    // get the subdetector names
    this->getSubdetectorNames();

    // create input selector
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    // declare the product to produce
    for (unsigned int ii=0;ii<simHitSubdetectors_.size();ii++) {
      produces<CrossingFrame<PSimHit> > (simHitSubdetectors_[ii]);
    }
    for (unsigned int ii=0;ii<caloSubdetectors_.size();ii++) {
      produces<CrossingFrame<PCaloHit> > (caloSubdetectors_[ii]);
    }
    produces<CrossingFrame<SimTrack> >();
    produces<CrossingFrame<SimVertex> >();
    produces<CrossingFrame<edm::HepMCProduct> >();
    
    //
    produces<CrossingFramePlaybackInfo>();  //FIXME: dependent on existence of rndmstore?
  }

  void MixingModule::getSubdetectorNames() {
    // get subdetector names
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.
      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName_.compare(0,9,"PCaloHits")) {
	caloSubdetectors_.push_back(desc.productInstanceName_);
	LogInfo("Constructor") <<"Adding calo container "<<desc.productInstanceName_ <<" for mixing";
      }
      else if (!desc.friendlyClassName_.compare(0,8,"PSimHits") && desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	simHitSubdetectors_.push_back(desc.productInstanceName_);
	nonTrackerPids_.push_back(desc.productInstanceName_);
        LogInfo("MixingModule") <<"Adding non tracker simhit container "<<desc.productInstanceName_ <<" for mixing";
      }
      else if (!desc.friendlyClassName_.compare(0,8,"PSimHits") && !desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	simHitSubdetectors_.push_back(desc.productInstanceName_);
	// here we store the tracker subdetector name  for low and high part
	int slow=(desc.productInstanceName_).find("LowTof");
	int iend=(desc.productInstanceName_).size();
        if (slow>0) {
 	  trackerHighLowPids_.push_back(desc.productInstanceName_.substr(0,iend-6));
	  LogInfo("MixingModule") <<"Adding tracker simhit container "<<desc.productInstanceName_.substr(0,iend-6) <<" for mixing";
        }
      }
    }
  }
  void MixingModule::beginJob(edm::EventSetup const&iSetup) {
  }

  void MixingModule::createnewEDProduct() {
    for (unsigned int ii=0;ii<simHitSubdetectors_.size();ii++) {
      cfSimHits_[simHitSubdetectors_[ii]]=new CrossingFrame<PSimHit>(minBunch_,maxBunch_,bunchSpace_,simHitSubdetectors_[ii],maxNbSources_);
    }
    for (unsigned int ii=0;ii<caloSubdetectors_.size();ii++) {
      cfCaloHits_[caloSubdetectors_[ii]]=new CrossingFrame<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,caloSubdetectors_[ii],maxNbSources_);
    }

    cfTracks_=new CrossingFrame<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(" "),maxNbSources_);
    cfVertices_=new CrossingFrame<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(" "),maxNbSources_);
    cfHepMC_=new CrossingFrame<edm::HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(" "),maxNbSources_);

    //create playback info
    playbackInfo_=new CrossingFramePlaybackInfo(minBunch_,maxBunch_,maxNbSources_); 
  }
 

  // Virtual destructor needed.
  MixingModule::~MixingModule() { 
    delete sel_;
  }  

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame

    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();
    std::map<std::string,CrossingFrame<PSimHit> * >::iterator it;

    // SimHits
    std::vector<edm::Handle<std::vector<PSimHit> > > resultsim;
    e.getMany((*sel_),resultsim);
    int ss=resultsim.size();

    for (int ii=0;ii<ss;ii++) {
      edm::BranchDescription desc = resultsim[ii].provenance()->product();
      LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<", "<<resultsim[ii].product()->size()<<" signal Simhits to be added";
      cfSimHits_[desc.productInstanceName_]->addSignals(resultsim[ii].product(),e.id());
    }

    // CaloHits
    std::map<std::string,CrossingFrame<PCaloHit> * >::iterator it2;

    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e.getMany((*sel_),resultcalo);
    int sc=resultcalo.size();

    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product();
      LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<", "<<resultcalo[ii].product()->size()<<" signal Calohits to be added";
      cfCaloHits_[desc.productInstanceName_]->addSignals(resultcalo[ii].product(),e.id());
    }

    //tracks and vertices
    std::vector<edm::Handle<std::vector<SimTrack> > > result_t;
    e.getMany((*sel_),result_t);
    int str=result_t.size();
    if (str>1) LogWarning("MixingModule") << " Found "<<str<<" SimTrack collections in signal file, only first one will be stored!!!!!!";
    if (str>0) {
      edm::BranchDescription desc =result_t[0].provenance()->product();
      LogDebug("MixingModule") <<" adding " << result_t[0].product()->size()<<" signal SimTracks";
      cfTracks_->addSignals(result_t[0].product(),e.id());
    }

    std::vector<edm::Handle<std::vector<SimVertex> > > result_v;
    e.getMany((*sel_),result_v);
    int sv=result_v.size();
    if (sv>1) LogWarning("MixingModule") << " Found "<<sv<<" SimVertex collections in signal file, only first one will be stored!!!!!!";
    if (sv>0) {
      LogDebug("MixingModule") <<" adding " << result_v[0].product()->size()<<" signal Simvertices ";
      cfVertices_->addSignals(result_v[0].product(),e.id());
    }
    
    //HepMC - we are creating a dummy vector, to have the same storage (MixCollection!) + interfaces
    std::vector<edm::Handle<edm::HepMCProduct> > result_mc;
    e.getMany((*sel_),result_mc);
    int smc=result_mc.size();
    if (smc>1) LogWarning("MixingModule") << " Found "<<smc<<" HepMCProducte in signal file, only first one will be stored!!!!!!";
    if (smc>0) {
      LogDebug("MixingModule") <<" adding signal HepMCProduct";
      std::vector<edm::HepMCProduct> vec;
      vec.push_back(*(result_mc[0].product()));
      cfHepMC_->addSignals(&vec,e.id());
    }
  }

  void MixingModule::addPileups(const int bcr, Event *e, unsigned int eventNr) {
  
    LogDebug("MixingModule") <<"\n===============> adding objects from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // SimHits
    // we have to treat tracker/non tracker  containers separately, prepare a global map
    // (all this due to the fact that we need to use getmany to avoid exceptions)
    std::map<const std::string,const std::vector<PSimHit>* > simproducts;
    std::vector<edm::Handle<std::vector<PSimHit> > > resultsim;
    e->getMany((*sel_),resultsim);
    int ss=resultsim.size();
    for (int ii=0;ii<ss;ii++) {
      edm::BranchDescription desc = resultsim[ii].provenance()->product();
      simproducts.insert(std::map<const std::string,const std::vector<PSimHit>* >::value_type(desc.productInstanceName_, resultsim[ii].product()));
    }

    // Non-tracker treatment
    for(std::vector <std::string>::iterator it = nonTrackerPids_.begin(); it != nonTrackerPids_.end(); ++it) {
      const std::vector<PSimHit> * simhits = simproducts[(*it)];
      if (simhits) {
	if (simhits->size()) {
	  cfSimHits_[(*it)]->addPileups(bcr,simhits,eventNr);
	  LogDebug("MixingModule") <<"For "<<(*it)<<", "<<simhits->size()<<"  Simhits added";
	}
      }
    }

    // Tracker treatment
    for(std::vector <std::string >::iterator itstr = trackerHighLowPids_.begin(); itstr != trackerHighLowPids_.end(); ++itstr) {
      const std::string subdethigh=(*itstr)+"HighTof";
      const std::string subdetlow=(*itstr)+"LowTof";

      // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
      // add HighTof simhits to high and low signals
      float tof = bcr*cfSimHits_[subdethigh]->getBunchSpace();
      if ( !checktof_ || ((limHighLowTof +tof ) <= highTrackTof)) { 
	const std::vector<PSimHit> * simhitshigh = simproducts[subdethigh];
	if (simhitshigh) {
          cfSimHits_[subdethigh]->addPileups(bcr,simhitshigh,eventNr,0,checktof_,true);
	  cfSimHits_[subdetlow]->addPileups(bcr,simhitshigh,eventNr,0,checktof_,false);
	  LogDebug("MixingModule") <<"For "<<subdethigh<<" + "<<subdetlow<<", "<<simhitshigh->size()<<" Hits added to high+low";
	}
      }

      // add LowTof simhits to high and low signals
      if (  !checktof_ || ((tof+limHighLowTof) >= lowTrackTof && tof <= highTrackTof)) {
	const std::vector<PSimHit> * simhitslow = simproducts[subdetlow];
	if (simhitslow) {
	  LogDebug("MixingModule") <<"For "<<subdethigh<<" + "<<subdetlow<<", "<<simhitslow->size()<<" Hits added to high+low";
	  cfSimHits_[subdethigh]->addPileups(bcr,simhitslow,eventNr,0,checktof_,true);
	  cfSimHits_[subdetlow]->addPileups(bcr,simhitslow,eventNr,0,checktof_,false);
	}
      }
    }

    // calo hits for all subdetectors
    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e->getMany((*sel_),resultcalo);
    int sc=resultcalo.size();
    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product();
      if (resultcalo[ii].product()->size()) {
	LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<" "<<resultcalo[ii].product()->size()<<" Calohits added";
	cfCaloHits_[desc.productInstanceName_]->addPileups(bcr,resultcalo[ii].product(),eventNr);
      }
    }

    //     //tracks and vertices
    std::vector<edm::Handle<std::vector<SimTrack> > > result_t;
    e->getMany((*sel_),result_t);
    int str=result_t.size();
    if (str>1) LogWarning("MixingModule") <<"Too many SimTrack containers, should be only one!";
    if (str>0) {
      LogDebug("MixingModule") <<result_t[0].product()->size()<<"  Simtracks added, eventNr "<<eventNr;
      if (result_t[0].isValid()) {
	cfTracks_->addPileups(bcr,result_t[0].product(),eventNr,vertexoffset);
      }
      else  LogWarning("MixingModule") <<"Invalid simtracks!";
    }

    std::vector<edm::Handle<std::vector<SimVertex> > > result_v;
    e->getMany((*sel_),result_v);
    int sv=result_v.size();
    if (sv>1) LogWarning("MixingModule") <<"Too many SimVertex containers, should be only one!"; 
    if (sv>0) {
      LogDebug("MixingModule") <<result_v[0].product()->size()<<"  Simvertices added";
      if (result_v[0].isValid()) {
	cfVertices_->addPileups(bcr,result_v[0].product(),eventNr);
      }
      else  LogWarning("MixingModule") <<"Invalid simvertices!";
      vertexoffset+=result_v[0].product()->size();
    }

    //HepMCProduct
    //HepMC - we are creating a dummy vector, to have the same interfaces
    std::vector<edm::Handle<edm::HepMCProduct> > result_mc;
    e->getMany((*sel_),result_mc);
    int smc=result_mc.size();
    if (smc>1) LogWarning("MixingModule") <<"Too many HepMCProducts, should be only one!"; 
    if (smc>0) {
      LogDebug("MixingModule") <<"  HepMCProduct added";
      std::vector<edm::HepMCProduct> vec;
      vec.push_back(*(result_mc[0].product()));
      cfHepMC_->addPileups(bcr,&vec,eventNr);
    }
  }

  void MixingModule::setBcrOffset() {
    for (unsigned int ii=0;ii<simHitSubdetectors_.size();ii++) {
      cfSimHits_[simHitSubdetectors_[ii]]->setBcrOffset();
    }
    for (unsigned int ii=0;ii<caloSubdetectors_.size();ii++) {
      cfCaloHits_[caloSubdetectors_[ii]]->setBcrOffset();
    }
    cfTracks_->setBcrOffset();
    cfVertices_->setBcrOffset();
    cfHepMC_->setBcrOffset();
  }

  void MixingModule::setEventStartInfo(const unsigned int s) {
   playbackInfo_->setEventStartInfo(eventIDs_,fileSeqNrs_,nrEvents_,s); 
  }

  void MixingModule::setSourceOffset(const unsigned int is) {
    for (unsigned int ii=0;ii<simHitSubdetectors_.size();ii++) {
      cfSimHits_[simHitSubdetectors_[ii]]->setSourceOffset(is);
    }
    for (unsigned int ii=0;ii<caloSubdetectors_.size();ii++) {
      cfCaloHits_[caloSubdetectors_[ii]]->setSourceOffset(is);
    }
    cfTracks_->setSourceOffset(is);
    cfVertices_->setSourceOffset(is);
    cfHepMC_->setSourceOffset(is);
  }

  void MixingModule::put(edm::Event &e) {
    std::map<std::string,CrossingFrame<PSimHit> * >::iterator it;
    for (it=cfSimHits_.begin();it!=cfSimHits_.end();it++) {
      std::auto_ptr<CrossingFrame<PSimHit> > pOut((*it).second);
      e.put(pOut,(*it).first);
    }
    std::map<std::string,CrossingFrame<PCaloHit> * >::iterator it2;
    for (it2=cfCaloHits_.begin();it2!=cfCaloHits_.end();it2++) {
      std::auto_ptr<CrossingFrame<PCaloHit> > pOut((*it2).second);
      e.put(pOut,(*it2).first);
    }
    if (cfTracks_) {
      e.put(std::auto_ptr<CrossingFrame<SimTrack> >(cfTracks_));
    }
    if (cfVertices_) {
      std::auto_ptr<CrossingFrame<SimVertex> > pOut(cfVertices_);
      e.put(pOut);
    }
    if (cfHepMC_) {
      std::auto_ptr<CrossingFrame<edm::HepMCProduct> > pOut(cfHepMC_);
      e.put(pOut);
    }
  
    if (playbackInfo_) {
      std::auto_ptr<CrossingFramePlaybackInfo> pOut(playbackInfo_);
      e.put(pOut);
    }
  }
  void MixingModule::getEventStartInfo(edm::Event & e, const unsigned int s) {
    // read event start info from event
    // and set it in BMixingModule
    //    id_=EventID(0,0);
    //    fileNr_=-1;
    if (playback_) {
 
      edm::Handle<CrossingFramePlaybackInfo>  playbackInfo_H;
      bool got=e.get((*sel_), playbackInfo_H);
      if (got) {
	playbackInfo_H->getEventStartInfo(eventIDs_,fileSeqNrs_,nrEvents_,s);
      }else{
	LogWarning("MixingModule")<<"\n\nAttention: No CrossingFramePlaybackInfo on the input file, but playback option set!!!!!!!\nAttention: Job is executed without playback, please change the input file if you really want playback!!!!!!!";
	//FIXME: defaults
      }
    }
  }
}//edm
