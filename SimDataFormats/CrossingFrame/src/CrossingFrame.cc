#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace std;
using namespace edm;

const int  CrossingFrame::lowTrackTof = -36;
const int  CrossingFrame::highTrackTof = 36;
const int  CrossingFrame::minLowTof =0;
const int  CrossingFrame::limHighLowTof =36;


CrossingFrame::CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> muonsubdetectors,std::vector<std::string> trackersubdetectors, std::vector<std::string> calosubdetectors): bunchSpace_(bunchsp), firstCrossing_(minb), lastCrossing_(maxb) {

    // create and insert vectors into the pileup map

//     // for muons
    for(std::vector<std::string >::const_iterator it = muonsubdetectors.begin(); it != muonsubdetectors.end(); ++it) {  
      vector<PSimHitContainer> myvec(-minb+maxb+1);
      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it),myvec));
     }

    // for tracker (hightof/lowtof are created as they come)

    for(std::vector<std::string >::const_iterator it = trackersubdetectors.begin(); it != trackersubdetectors.end(); ++it) {  
      vector<PSimHitContainer> myvec(-minb+maxb+1);
      //      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it)+"HighTof",myvec));    

      //      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it)+"LowTof",myvec));
      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it),myvec));
     }

    // for calos
    for(vector<string >::const_iterator it = calosubdetectors.begin(); it != calosubdetectors.end(); ++it) {  
      vector<PCaloHitContainer> myvec(-minb+maxb+1);
      pileupCaloHits_.insert(map <string, vector<PCaloHitContainer> >::value_type((*it),myvec));
    }

    // fill vectors for track and vertex pileup
    pileupTracks_.resize(-minb+maxb+1);
    pileupVertices_.resize(-minb+maxb+1);
  }

  CrossingFrame::~CrossingFrame () {
    this->clear();
  }

void CrossingFrame::clear() {
  // clear things up
  signalSimHits_.clear();
  signalCaloHits_.clear();
  signalTracks_.clear();
  signalVertices_.clear();
  pileupSimHits_.clear();
  pileupCaloHits_.clear();
  pileupTracks_.clear();
  pileupVertices_.clear();
}

void CrossingFrame::addSignalSimHits(const std::string subdet, const PSimHitContainer *simhits) { 
  signalSimHits_.insert(map <string, PSimHitContainer>::value_type(subdet,*simhits));
}

void CrossingFrame::addSignalCaloHits(const std::string subdet, const PCaloHitContainer *calohits) { 
  signalCaloHits_.insert(map <string, PCaloHitContainer>::value_type(subdet,*calohits));
}

void CrossingFrame::addSignalTracks(const EmbdSimTrackContainer *simtracks) { 
  signalTracks_=*simtracks;
}

void CrossingFrame::addSignalVertices(const EmbdSimVertexContainer *simvertices) { 
  signalVertices_=*simvertices;
}


void CrossingFrame::addPileupSimHits(const int bcr, const std::string subdet, const PSimHitContainer *simhits, int trackoffset, bool checkTof) { 
  // add individual PSimHits to this bunchcrossing
  // eliminate those which a TOF outside of the bounds to be considered for corresponding detectors only

  int count=0;
  for (unsigned int i=0;i<simhits->size();++i) {
    bool accept=true;
    float newtof;
    if (checkTof) {
      newtof=(*simhits)[i].timeOfFlight() + bcr*bunchSpace_;
      accept=newtof>=lowTrackTof && newtof <=highTrackTof;
    }
    if (!checkTof || accept) {
      PSimHit hit((*simhits)[i].entryPoint(), (*simhits)[i].exitPoint(),(*simhits)[i].pabs(),
		  (*simhits)[i].timeOfFlight() + bcr*bunchSpace_, 
		  (*simhits)[i].energyLoss(), (*simhits)[i].particleType(),
		  (*simhits)[i].detUnitId(), (*simhits)[i].trackId()+trackoffset,
		  (*simhits)[i].thetaAtEntry(),  (*simhits)[i].phiAtEntry(),  (*simhits)[i].processType());
      (pileupSimHits_[subdet])[bcr-firstCrossing_].push_back(hit);
      count++;
    }  
  }
}

  void CrossingFrame::addPileupCaloHits(const int bcr, const std::string subdet, const PCaloHitContainer *calohits, int trackoffset) { 
    for (unsigned int i=0;i<calohits->size();++i) {
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energy(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId()+trackoffset);
    (pileupCaloHits_[subdet])[bcr-firstCrossing_].push_back(hit);
  }
}

void CrossingFrame::addPileupTracks(const int bcr, const EmbdSimTrackContainer *simtracks, int vertexoffset) { 
  for (unsigned int i=0;i<simtracks->size();++i) 
    if ((*simtracks)[i].noVertex()) 
      //      pileupTracks_[bcr-firstCrossing_].insertTrack((*simtracks)[i]);
      pileupTracks_[bcr-firstCrossing_].push_back((*simtracks)[i]);
    else {
      EmbdSimTrack track((*simtracks)[i].type(),(*simtracks)[i].momentum(),(*simtracks)[i].vertIndex()+vertexoffset, (*simtracks)[i].genpartIndex());
      //      pileupTracks_[bcr-firstCrossing_].insertTrack(track);
      pileupTracks_[bcr-firstCrossing_].push_back(track);
    }
}

void CrossingFrame::addPileupVertices(const int bcr, const EmbdSimVertexContainer *simvertices, int trackoffset) { 
  for (unsigned int i=0;i<simvertices->size();++i) 
    if ((*simvertices)[i].noParent()) 
      //      pileupVertices_[bcr-firstCrossing_].insertVertex((*simvertices)[i]);
      pileupVertices_[bcr-firstCrossing_].push_back((*simvertices)[i]);
    else {
      EmbdSimVertex vertex((*simvertices)[i].position(),((*simvertices)[i].position())[3]+bcr*bunchSpace_,(*simvertices)[i].parentIndex()+trackoffset);
      //      pileupVertices_[bcr-firstCrossing_].insertVertex(vertex);
      pileupVertices_[bcr-firstCrossing_].push_back(vertex);
    }
}

void CrossingFrame::print(int level) const {

  std::cout<<*this<<std::endl;
  //"\nCrossingFrame for "<<id_<<",  minbunch = "<<firstCrossing_
  //	   <<", bunchSpace "<<bunchSpace_<<std::endl;

    // print for lowest level (signals)
  //
  // signals
  cout<<"\nSignals:"<<endl;
  for(map<string,PSimHitContainer>::const_iterator it = signalSimHits_.begin(); it != signalSimHits_.end(); ++it) {
    cout<< " subdetector "<<(*it).first <<" signal size "<<(*it).second.size()<<endl;
    if (level>=2) {
      for (unsigned int j=0;j<(*it).second.size();++j) {
        cout<<" SimHit "<<j<<" has track pointer "<< (*it).second[j].trackId() <<" ,tof "<<(*it).second[j].tof()<<", energy loss "<< (*it).second[j].energyLoss()<<endl;
      }
    }
  }

  for(map<string,PCaloHitContainer>::const_iterator it = signalCaloHits_.begin(); it != signalCaloHits_.end(); ++it) {
    cout<< " subdetector "<<(*it).first <<" signal size "<<(*it).second.size()<<endl;
    if (level>=2) {
      for (unsigned int j=0;j<(*it).second.size();++j) {
        HcalDetId detid = (HcalDetId)(*it).second[j].id();
	cout << (*it).second[j]  << ", detid: "<< detid << endl;
//	cout<<" CaloHit "<<j<<" has track pointer "<< (*it).second[j].geantTrackId() <<" ,tof "<<(*it).second[j].time()<<", energy loss "<< (*it).second[j].energy()<<endl;
      }
    }
  }

  cout <<" Number of tracks in signal "<< signalTracks_.size()<<endl;
  if (level>=2) {
    for (unsigned int j=0;j<signalTracks_.size();++j) 
      cout<<" track "<<j<<" has vertex pointer "<< signalTracks_[j].vertIndex()<<" and genpartindex "<<signalTracks_[j].genpartIndex()<<endl;
  }
  cout <<" Number of vertices in signal "<< signalVertices_.size()<<endl;
  	if (level>=2) {
  	  for (unsigned int j=0;j<signalVertices_.size();++j) 
   	    cout<<" vertex "<<j<<" has track pointer "<< signalVertices_[j].parentIndex()<<endl;
  	}

  
	//  print for next higher level (pileups)
  
  if (level<1) return;
  cout<<"\nPileups:"<<endl;
  map<string,vector<PSimHitContainer> >::const_iterator itsim;
  for(itsim = pileupSimHits_.begin(); itsim != pileupSimHits_.end(); ++itsim){ 
    cout<< endl<<" Pileup for subdetector "<<(*itsim).first <<endl;
    for (unsigned int i=0;i<(*itsim).second.size();++i) {
      int bcr=firstCrossing_+i;
      cout <<" Bunchcrossing  "<<bcr<<", Simhit pileup size "<<(*itsim).second[i].size()<<endl;
      if (level>=3) {
	for (unsigned int j=0;j<(*itsim).second[i].size();++j) 
	  cout<<" SimHit "<<j<<" has track pointer "<< ((*itsim).second[i])[j].trackId() <<" ,tof "<<((*itsim).second[i])[j].tof()<<", energy loss "<< ((*itsim).second[i])[j].energyLoss()<<endl;
      }
    }
  }
  

  map<string,vector<PCaloHitContainer> >::const_iterator it;
  for(it = pileupCaloHits_.begin(); it != pileupCaloHits_.end(); ++it){ 
    cout<< " Pileup for subdetector "<<(*it).first <<endl;
    for (unsigned int i=0;i<(*it).second.size();++i) {
      int bcr=firstCrossing_+i;
      cout <<" Bunchcrossing  "<<bcr<<", Calohit pileup size "<<(*it).second[i].size()<<endl;
      if (level>=3) {
	for (unsigned int j=0;j<(*it).second[i].size();++j) {
	  //          cout<<" CaloHit "<<j<<" has track pointer "<< ((*it).second[i])[j].geantTrackId() <<" ,tof "<<((*it).second[i])[j].time()<<"energy "<< ((*it).second[i])[j].energy()<<endl;
	  HcalDetId detid = (HcalDetId)((*it).second[i])[j].id();
	  cout << ((*it).second[i])[j]  << ", detid: "<< detid << endl;
	}
      }
    }
  }

  for (unsigned int i=0;i<pileupTracks_.size();++i) {
    int bcr=firstCrossing_+i;
    cout <<" Bunchcrossing  "<<bcr<<", Nr  pileup tracks "<<pileupTracks_[i].size();
    if (level>=3) {
      cout<<endl;
      for (unsigned int j=0;j<pileupTracks_[i].size();++j) 
	cout<<" track "<<j<<" has vertex pointer "<< (pileupTracks_[i])[j].vertIndex()<<" and genpartindex "<<(pileupTracks_[i])[j].genpartIndex()<<endl;
    }
    cout<<", Nr  pileup vertices "<<pileupVertices_[i].size( )<<endl;
    // 	  if (level>=3) {
    // 	  for (unsigned int j=0;j<pileupVertices_[i].size();++j) 
    // 	    cout<<" vertex "<<j<<" has track pointer "<< (pileupVertices_[i])[j].parentIndex()<<endl;
    // 	  }
  }
   
}

std::ostream &operator<<(std::ostream& o, const CrossingFrame &cf)
{
  std::pair<int,int> range=cf.getBunchRange();
  o <<"\nCrossingFrame for "<<cf.getEventID()<<",  bunchrange = "<<range.first<<","<<range.second
	   <<", bunchSpace "<<cf.getBunchSpace();

  return o;
}


