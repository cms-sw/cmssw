// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "SimGeneral/MixingModule/interface/MixingModule.h"
#include "FWCore/Framework/interface/Handle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps) : BMixingModule(ps)
  {

    //temporary
    const char *labels[]={"TrackerHitsPixelBarrelHighTof"
			  ,"TrackerHitsPixelBarrelLowTof"
			  ,"TrackerHitsPixelEndcapHighTof"
			  ,"TrackerHitsPixelEndcapLowTof"
			  ,"TrackerHitsTECHighTof"
			  ,"TrackerHitsTECLowTof"
			  ,"TrackerHitsTIBHighTof"
			  ,"TrackerHitsTIBLowTof"
			  ,"TrackerHitsTIDHighTof"
			  ,"TrackerHitsTIDLowTof"
			  ,"TrackerHitsTOBHighTof"
			  ,"TrackerHitsTOBLowTof"
    };
    for (int i=0;i<12;++i)
      trackerSubdetectors_.push_back(std::string(labels[i]));

    produces<CrossingFrame> ();
     
  }

  void MixingModule::createnewEDProduct() {
    simcf_=new CrossingFrame(minbunch_,maxbunch_,bunchSpace_,trackerSubdetectors_,caloSubdetectors_);
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() { }  

  void MixingModule::addSignals(edm::Event &e) { 
    // fill in signal part of CrossingFrame
    // first add eventID
    simcf_->setEventID(e.id());

    // tracker hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = trackerSubdetectors_.begin(); it != trackerSubdetectors_.end(); ++it) {  
      edm::Handle<edm::PSimHitContainer> simHits;
      e.getByLabel("r",(*it),simHits);
      simcf_->addSignalSimHits((*it),simHits.product());
      cout <<" Got "<<(simHits.product())->size()<<" simhits for subdet "<<(*it)<<endl;
    }
//     // cal hits for all subdetectors
//     for(std::vector<std::string >::const_iterator it = caloSubdetectors_.begin(); it != caloSubdetectors_.end(); ++it) {  
//       edm::Handle<edm::PCaloHitContainer> caloHits;
//       e.getByLabel("r",(*it),caloHits);
//       simcf_->addSignalCaloHits((*it),caloHits.product());
//       cout <<" Got "<<(caloHits.product())->size()<<" calohits for subdet "<<(*it)<<endl;
//     }
    edm::Handle<edm::EmbdSimTrackContainer> simtracks;
    e.getByLabel("r",simtracks);
    if (simtracks.isValid()) simcf_->addSignalTracks(simtracks.product());
    else cout <<"Invalid simtracks"<<endl;
    cout <<" Got "<<(simtracks.product())->size()<<" simtracks"<<endl;
    edm::Handle<edm::EmbdSimVertexContainer> simvertices;
    e.getByLabel("r",simvertices);
    if (simvertices.isValid())     simcf_->addSignalVertices(simvertices.product());
    else cout <<"Invalid simvertices"<<endl;
    cout <<" Got "<<(simvertices.product())->size()<<" simvertices"<<endl;
  }

  void MixingModule::addPileups(const int bcr, Event *e) {

   // first all simhits
    for(std::vector<std::string >::const_iterator itstr = trackerSubdetectors_.begin(); itstr != trackerSubdetectors_.end(); ++itstr) {
      edm::Handle<edm::PSimHitContainer>  simHits;  //Event Pointer to minbias Hits
      e->getByLabel("r",(*itstr),simHits);
      simcf_->addPileupSimHits(bcr,(*itstr),simHits.product());
    }

//     //then all calohits
//     for(std::vector<std::string >::const_iterator itstr = caloSubdetectors_.begin(); itstr != caloSubdetectors_.end(); ++itstr) {
//       edm::Handle<edm::PCaloHitContainer>  caloHits;  //Event Pointer to minbias Hits
//       e->getByLabel("r",(*itstr),caloHits);
//       simcf_->addPileupCaloHits(bcr,(*itstr),caloHits.product());
//     }
    //then simtracks
    edm::Handle<edm::EmbdSimTrackContainer> simtracks;
    e->getByLabel("r",simtracks);
    if (simtracks.isValid()) simcf_->addPileupTracks(bcr, simtracks.product());
    else cout <<"Invalid simtracks"<<endl;

    //then simvertices
    edm::Handle<edm::EmbdSimVertexContainer> simvertices;
    e->getByLabel("r",simvertices);
    if (simvertices.isValid())  simcf_->addPileupVertices(bcr,simvertices.product());
    else cout <<"Invalid simvertices"<<endl;
  }
 
  void MixingModule::put(edm::Event &e) {
    e.put(std::auto_ptr<CrossingFrame>(simcf_));
  }

} //edm
