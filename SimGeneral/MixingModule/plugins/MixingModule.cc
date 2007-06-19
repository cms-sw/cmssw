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
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "MixingModule.h"


using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps) : BMixingModule(ps),
			     label_(ps.getParameter<std::string>("Label"))

  {
    // declare the product to produce
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
      produces<CrossingFrame> (label_);
    }
    else {
      sel_=new Selector( MatchAllSelector());
      produces<CrossingFrame> ();
    }
  }

  void MixingModule::beginJob(edm::EventSetup const&iSetup) {
    // get subdetector names
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.
      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName_.compare(0,8,"PCaloHit")) {
	caloSubdetectors_.push_back(desc.productInstanceName_);
	LogInfo("Constructor") <<"Adding calo container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.friendlyClassName_.compare(0,7,"PSimHit") && desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	simHitSubdetectors_.push_back(desc.productInstanceName_);
	nonTrackerPids_.push_back(desc.productInstanceName_);
        LogInfo("Constructor") <<"Adding simhit container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.friendlyClassName_.compare(0,7,"PSimHit") && !desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	simHitSubdetectors_.push_back(desc.productInstanceName_);
	// here we store the tracker subdetector name  for low and high part
	int slow=(desc.productInstanceName_).find("LowTof");
	int iend=(desc.productInstanceName_).size();
	if (slow>0) {
 	  trackerHighLowPids_.push_back(desc.productInstanceName_.substr(0,iend-6));
	  LogInfo("MixingModule") <<"Adding container "<<desc.productInstanceName_.substr(0,iend-6) <<" for pileup treatment";
        }
      }
      //      else
      //        cout<<"Strange detector "<<desc.productInstanceName_ <<",productID "<<desc.productID_<<" for pileup treatment????????"<<endl;
    }
  }

  void MixingModule::createnewEDProduct() {
    simcf_=new CrossingFrame(minBunch(),maxBunch(),bunchSpace_,simHitSubdetectors_,caloSubdetectors_);
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() { 
    delete sel_;
  }  

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame

    // first add eventID
    simcf_->setEventID(e.id());
    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();
    eventId_=0;

    // SimHits
    std::vector<edm::Handle<std::vector<PSimHit> > > resultsim;
    e.getMany((*sel_),resultsim);
    int ss=resultsim.size();
    if (ss>1) LogWarning("MixingModule") << " Found "<<ss<<" PSimHit collections in signal file, only first one  will be stored!!!!!!";
    for (int ii=0;ii<ss;ii++) {
      edm::BranchDescription desc = resultsim[ii].provenance()->product();
      LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<" "<<resultsim[ii].product()->size()<<" Simhits added";
      simcf_->addSignalSimHits(desc.productInstanceName_,resultsim[ii].product());
    }


    // calo hits for all subdetectors
    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e.getMany((*sel_),resultcalo);
    int sc=resultcalo.size();
    if (sc>1) LogWarning("MixingModule") << " Found "<<sc<<" PCaloHit collections in signal file, only first one  will be stored!!!!!!";
    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product();
      LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<" "<<resultcalo[ii].product()->size()<<" Calohits added";
      simcf_->addSignalCaloHits(desc.productInstanceName_,resultcalo[ii].product());
    }
  

//     //tracks and vertices
    std::vector<edm::Handle<std::vector<SimTrack> > > result_t;
    e.getMany((*sel_),result_t);
    int str=result_t.size();
    if (str>1) LogWarning("MixingModule") << " Found "<<str<<" SimTrack collections in signal file, only first one  will be stored!!!!!!";
    for (int ii=0;ii<str;ii++) {
      edm::BranchDescription desc =result_t[ii].provenance()->product();
      LogDebug("MixingModule") <<result_t[ii].product()->size()<<" Simtracks added";
      if (result_t[ii].isValid()) simcf_->addSignalTracks(result_t[ii].product());
      else  LogWarning("InvalidData") <<"Invalid simtracks in signal";
    }

    std::vector<edm::Handle<std::vector<SimVertex> > > result_v;
    e.getMany((*sel_),result_v);
    int sv=result_v.size();
    if (sv>1) LogWarning("MixingModule") << " Found "<<sv<<" SimTrack collections in signal file, only first one  will be stored!!!!!!";
    for (int ii=0;ii<sv;ii++) {
      edm::BranchDescription desc = result_v[ii].provenance()->product();
      LogDebug("MixingModule") <<result_v[ii].product()->size()<<" Simvertices added";
      if (result_v[ii].isValid()) simcf_->addSignalVertices(result_v[ii].product());
      else  LogWarning("InvalidData") <<"Invalid simvertices in signal";
    }
  }

  void MixingModule::addPileups(const int bcr, Event *e, unsigned int eventId) {
  
    LogDebug("MixingModule") <<"===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

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
	  simcf_->addPileupSimHits(bcr,(*it),simhits,eventId,false);
	  LogDebug("MixingModule") <<"For "<<(*it)<<", "<<simhits->size()<<" Simhits added";
	}
    }
    // Tracker treatment
    float tof = bcr*simcf_->getBunchSpace();
    for(std::vector <std::string >::iterator itstr = trackerHighLowPids_.begin(); itstr != trackerHighLowPids_.end(); ++itstr) {
      std::string subdethigh=(*itstr)+"HighTof";
      std::string subdetlow=(*itstr)+"LowTof";
      // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
      // add HighTof pileup to high and low signals
      if ( !checktof_ || ((CrossingFrame::limHighLowTof +tof ) <= CrossingFrame::highTrackTof)) { 

	const std::vector<PSimHit> * simhitshigh = simproducts[subdethigh];
	if (simhitshigh) {
	  simcf_->addPileupSimHits(bcr,subdethigh,simhitshigh,eventId,checktof_);
	  simcf_->addPileupSimHits(bcr,subdetlow,simhitshigh,eventId,checktof_);
	  LogDebug("MixingModule") <<"For "<<subdethigh<<" and "<<subdetlow<<", "<<simhitshigh->size()<<" Simhits added";
	}
      }

      // add LowTof pileup to high and low signals
      if (  !checktof_ || ((tof+CrossingFrame::limHighLowTof) >= CrossingFrame::lowTrackTof && tof <= CrossingFrame::highTrackTof)) {     
	//	const std::vector<PSimHit> * simhitslow = simproducts[(*itstr).second.second];
	const std::vector<PSimHit> * simhitslow = simproducts[subdetlow];
	if (simhitslow) {
	  simcf_->addPileupSimHits(bcr,subdethigh,simhitslow,eventId, checktof_);
	  simcf_->addPileupSimHits(bcr,subdetlow,simhitslow,eventId, checktof_);
	  LogDebug("MixingModule") <<"For "<<subdethigh<<" and "<<subdetlow<<", "<<simhitslow->size()<<" Simhits added";
	}
      }
    }

    // calo hits for all subdetectors
    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e->getMany((*sel_),resultcalo);
    int sc=resultcalo.size();
    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product();
      LogDebug("MixingModule") <<"For "<<desc.productInstanceName_<<" "<<resultcalo[ii].product()->size()<<" Calohits added";
      simcf_->addPileupCaloHits(bcr,desc.productInstanceName_,resultcalo[ii].product(),eventId);
    }
 
//     //tracks and vertices
    std::vector<edm::Handle<std::vector<SimTrack> > > result_t;
    e->getMany((*sel_),result_t);
    int str=result_t.size();
    if (str>1) LogWarning("InvalidData") <<"Too many SimTrack containers, should be only one!";
    for (int ii=0;ii<str;ii++) {
      edm::BranchDescription desc =result_t[ii].provenance()->product();
      LogDebug("MixingModule") <<result_t[ii].product()->size()<<" Simtracks added";
      if (result_t[ii].isValid()) simcf_->addPileupTracks(bcr,result_t[ii].product(),eventId,vertexoffset);
      else  LogWarning("InvalidData") <<"Invalid simtracks in signal";
    }
  

    std::vector<edm::Handle<std::vector<SimVertex> > > result_v;
    e->getMany((*sel_),result_v);
    int sv=result_v.size();
    if (sv>1) LogWarning("InvalidData") <<"Too many SimVertex containers, should be only one!";
    for (int ii=0;ii<sv;ii++) {
      edm::BranchDescription desc = result_v[ii].provenance()->product();
      LogDebug("MixingModule") <<result_v[ii].product()->size()<<" Simvertices added";
      if (result_v[ii].isValid()) simcf_->addPileupVertices(bcr,result_v[ii].product(),eventId);
      else  LogWarning("InvalidData") <<"Invalid simvertices in signal";
      if (ii==1) vertexoffset+=result_v[ii].product()->size();
    }

  }
 
  void MixingModule::put(edm::Event &e) {
    e.put(std::auto_ptr<CrossingFrame>(simcf_),label_);
  }

} //edm
