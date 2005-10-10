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
    printf("===================>>>> createnewEDProduct called \n");fflush(stdout);
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

  void MixingModule::getEvents (const unsigned int nrEvents)
  {
    // filling of eventvector by using  secondary input source 
    // temporary for workarounds, in the future it will be implemented in the baseclass only
    unsigned int eventCount=0;
    eventVector_.clear();
    std::vector <edm::Handle<edm::PSimHitContainer> > simHits;  //Event Pointer to minbias Hits
    std::vector <edm::Handle<edm::PCaloHitContainer> > caloHits;  //Event Pointer to minbias Hits
    std::vector<EventPrincipal*> vecEventPrincipal;
    secInput_->readMany(0, nrEvents, vecEventPrincipal);
    while (eventCount < nrEvents) {
      ModuleDescription md=ModuleDescription();  //temporary
      Event *event = new Event(*vecEventPrincipal[eventCount], md);
      cout <<"\n Pileup event nr "<<eventCount<<" event id "<<event->id()<<endl;
      eventVector_.push_back (event);
      // Force reading of the hits now!!!!
      event->getManyByType(simHits);  // Workaround
      for (unsigned int idet=0;idet<simHits.size();idet++) 
	cout <<" Got "<<(simHits[idet].product())->size()<<" PILEUP simhits for subdet "<<idet<<endl;
      simHits.clear(); //Workaround
      //      event->getManyByType(caloHits);  // Workaround
      //      caloHits.clear(); //Workaround

      edm::Handle<edm::EmbdSimTrackContainer> simtracks; // Workaround
      edm::Handle<edm::EmbdSimVertexContainer> simvertices; // Workaround

      event->getByLabel("r",simtracks); //Workaround
      event->getByLabel("r",simvertices); //Workaround
      cout <<" Got "<<(simtracks.product())->size()<<"PILEUP simtracks"<<endl;
      cout <<" Got "<<(simvertices.product())->size()<<" PILEUP simvertices"<<endl;
      ++eventCount;
    }
    cout <<endl;
  }
 

} //edm
