#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace std;
using namespace edm;

const int  CrossingFrame::lowTrackTof = -36;
const int  CrossingFrame::highTrackTof = 36;
const int  CrossingFrame::minLowTof =0;
const int  CrossingFrame::limHighLowTof =36;


CrossingFrame::CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> simHitSubdetectors, std::vector<std::string> caloSubdetectors): bunchSpace_(bunchsp), firstCrossing_(minb), lastCrossing_(maxb) {

    // create and insert vectors into the pileup map

//     // for simhits
    for(std::vector<std::string >::const_iterator it = simHitSubdetectors.begin(); it != simHitSubdetectors.end(); ++it) {  
      vector<PSimHitContainer> myvec(-minb+maxb+1);
      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it),myvec));
     }

//     // for tracker (hightof/lowtof are created as they come)

//     for(std::vector<std::string >::const_iterator it = trackersubdetectors.begin(); it != trackersubdetectors.end(); ++it) {  
//       vector<PSimHitContainer> myvec(-minb+maxb+1);
//       pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it)+"HighTof",myvec));    

//       pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it)+"LowTof",myvec));
//       //      pileupSimHits_.insert(map <string, vector<PSimHitContainer> >::value_type((*it),myvec));
//      }

    // for calos
    for(vector<string >::const_iterator it = caloSubdetectors.begin(); it != caloSubdetectors.end(); ++it) {  
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

void CrossingFrame::addSignalTracks(const SimTrackContainer *simtracks) { 
  signalTracks_=*simtracks;
}

void CrossingFrame::addSignalVertices(const SimVertexContainer *simvertices) { 
  signalVertices_=*simvertices;
}


void CrossingFrame::addPileupSimHits(const int bcr, const std::string subdet, const PSimHitContainer *simhits, int evtId, bool checkTof) { 
  // add individual PSimHits to this bunchcrossing
  // eliminate those which a TOF outside of the bounds to be considered for corresponding detectors only

  int count=0;
  EncodedEventId id(bcr,evtId);
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
		  (*simhits)[i].detUnitId(), (*simhits)[i].trackId(),
		  (*simhits)[i].thetaAtEntry(),  (*simhits)[i].phiAtEntry(),  (*simhits)[i].processType());
      hit.setEventId(id);
      (pileupSimHits_[subdet])[bcr-firstCrossing_].push_back(hit);
      count++;
    }  
  }
}

//shifted version:  void CrossingFrame::addPileupCaloHits(const int bcr, const std::string subdet, const PCaloHitContainer *calohits, int trackoffset) { 
 void CrossingFrame::addPileupCaloHits(const int bcr, const std::string subdet, const PCaloHitContainer *calohits) { 
    for (unsigned int i=0;i<calohits->size();++i) {
      //shifted version:    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energy(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId()+trackoffset);
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energy(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId());
    (pileupCaloHits_[subdet])[bcr-firstCrossing_].push_back(hit);
  }
}

void CrossingFrame::addPileupTracks(const int bcr, const SimTrackContainer *simtracks, int evtId, int vertexoffset) { 
  EncodedEventId id(bcr,evtId);
  for (unsigned int i=0;i<simtracks->size();++i)
    if ((*simtracks)[i].noVertex()) {
      SimTrack track((*simtracks)[i]);
      track.setEventId(id);
      pileupTracks_[bcr-firstCrossing_].push_back(track);
    }
    else {
      SimTrack track((*simtracks)[i].type(),(*simtracks)[i].momentum(),(*simtracks)[i].vertIndex()+vertexoffset, (*simtracks)[i].genpartIndex());
      track.setEventId(id);
      pileupTracks_[bcr-firstCrossing_].push_back(track);
    }
}

void CrossingFrame::addPileupVertices(const int bcr, const SimVertexContainer *simvertices, int evtId,  int trackoffset) { 
    for (unsigned int i=0;i<simvertices->size();++i) 
    if ((*simvertices)[i].noParent()) {
      SimVertex vertex((*simvertices)[i]);
      vertex.setEventId(EncodedEventId(bcr,evtId));
      pileupVertices_[bcr-firstCrossing_].push_back(vertex);
    }
    else {
      SimVertex vertex((*simvertices)[i].position(),((*simvertices)[i].position())[3]+bcr*bunchSpace_,(*simvertices)[i].parentIndex()+trackoffset);
      vertex.setEventId(EncodedEventId(bcr,evtId));
      pileupVertices_[bcr-firstCrossing_].push_back(vertex);
    }
}

// version without shift
void CrossingFrame::addPileupVertices(const int bcr, const SimVertexContainer *simvertices, int evtId) { 
  for (unsigned int i=0;i<simvertices->size();++i){ 
      SimVertex vertex((*simvertices)[i]);
      vertex.setEventId(EncodedEventId(bcr,evtId));
      pileupVertices_[bcr-firstCrossing_].push_back(vertex);
  }
}

void CrossingFrame::print(int level) const {
  //FIXME to be corrected for higher levels
  std::cout<<*this<<std::endl;
  //
  // first, give a summary
  cout<<"\nLoop over all detectors:"<<endl;
  for (map<string,std::vector<PSimHitContainer> >::const_iterator it = pileupSimHits_.begin(); it != pileupSimHits_.end(); ++it) {
    int sig=0;
    const std::string sub=(*it).first;
    //    PSimHitContainer signals=signalSimHits_[sub];
    //    sig=sig+signals.size();
    //cout<< " Subdetector "<<(*it).first <<", signal size "<<signalSimHits_[(*it).first].size();
    //??    cout<< " Subdetector "<<(*it).first <<", signal size "<<(signalSimHits_[sub]).size();
    int pileupHits=0;
    for (unsigned int j=0;j<(*it).second.size();++j)      pileupHits=pileupHits+((*it).second)[j].size() ;
    cout<< " Subdetector "<<(*it).first<<", pileup size "<<pileupHits<<endl;
  }
  
  
  cout <<"\n Number of tracks in signal: "<< signalTracks_.size()<<endl;
  cout <<" Number of vertices in signal: "<< signalVertices_.size()<<endl;
  if (level<=1) return;

  //   for(map<string,PSimHitContainer>::const_iterator it = signalSimHits_.begin(); it != signalSimHits_.end(); ++it) {
//     cout<< " Subdetector "<<(*it).first <<", signal size "<<(*it).second.size()<<endl;
//     if (level>=2) {
//       for (unsigned int j=0;j<(*it).second.size();++j) {
//         cout<<" SimHit "<<j<<" has track pointer "<< (*it).second[j].trackId() <<" ,tof "<<(*it).second[j].tof()<<", energy loss "<< (*it).second[j].energyLoss()<<endl;
//       }
//     }
//   }

//   for(map<string,PCaloHitContainer>::const_iterator it = signalCaloHits_.begin(); it != signalCaloHits_.end(); ++it) {
//     cout<< " Subdetector "<<(*it).first <<", signal size "<<(*it).second.size()<<endl;
//     if (level>=2) {
//       for (unsigned int j=0;j<(*it).second.size();++j) {
//         HcalDetId detid = (HcalDetId)(*it).second[j].id();
// 	cout << (*it).second[j]  << ", detid: "<< detid << endl;
// //	cout<<" CaloHit "<<j<<" has track pointer "<< (*it).second[j].geantTrackId() <<" ,tof "<<(*it).second[j].time()<<", energy loss "<< (*it).second[j].energy()<<endl;
//       }
//     }
//   }

//   cout <<" Number of tracks in signal: "<< signalTracks_.size()<<endl;
//   if (level>=2) {
//     for (unsigned int j=0;j<signalTracks_.size();++j) 
//       cout<<" track "<<j<<" has vertex pointer "<< signalTracks_[j].vertIndex()<<" and genpartindex "<<signalTracks_[j].genpartIndex()<<endl;
//   }
//   cout <<" Number of vertices in signal: "<< signalVertices_.size()<<endl;
//   	if (level>=2) {
//   	  for (unsigned int j=0;j<signalVertices_.size();++j) 
//    	    cout<<" vertex "<<j<<" has track pointer "<< signalVertices_[j].parentIndex()<<endl;
//   	}

  
	//  print for next higher level (pileups)
  
//   if (level<1) return;
//   cout<<"\nPileups:"<<endl;
//   map<string,vector<PSimHitContainer> >::const_iterator itsim;
//   for(itsim = pileupSimHits_.begin(); itsim != pileupSimHits_.end(); ++itsim){ 
//     cout<< endl<<" Nr Hits for "<<(*itsim).first<<":";
//     for (unsigned int i=0;i<(*itsim).second.size();++i) {
//       int bcr=firstCrossing_+i;
//       cout <<" bcr="<<bcr<<": "<<(*itsim).second[i].size()<<", ";
// //       if (level>=3) {
// // 	for (unsigned int j=0;j<(*itsim).second[i].size();++j) 
// // 	  cout<<" SimHit "<<j<<" has track pointer "<< ((*itsim).second[i])[j].trackId() <<" ,tof "<<((*itsim).second[i])[j].tof()<<", energy loss "<< ((*itsim).second[i])[j].energyLoss()<<endl;
// //       }
//     }
//     cout <<endl;
//   }
  

//   map<string,vector<PCaloHitContainer> >::const_iterator it;
//   for(it = pileupCaloHits_.begin(); it != pileupCaloHits_.end(); ++it){ 
//     cout<< endl<<" Nr Hits for "<<(*it).first<<":";
//     for (unsigned int i=0;i<(*it).second.size();++i) {
//       int bcr=firstCrossing_+i;
//        cout <<" bcr="<<bcr<<": "<<(*it).second[i].size()<<", ";
// //       if (level>=3) {
// // 	for (unsigned int j=0;j<(*it).second[i].size();++j) {
// // 	  //          cout<<" CaloHit "<<j<<" has track pointer "<< ((*it).second[i])[j].geantTrackId() <<" ,tof "<<((*it).second[i])[j].time()<<"energy "<< ((*it).second[i])[j].energy()<<endl;
// // 	  HcalDetId detid = (HcalDetId)((*it).second[i])[j].id();
// // 	  cout << ((*it).second[i])[j]  << ", detid: "<< detid << endl;
// // 	}
// //       }
//     }
//     cout<<endl;
//   }

//   cout <<"\n Tracks "<<endl;
//   for (unsigned int i=0;i<pileupTracks_.size();++i) {
//     int bcr=firstCrossing_+i;
//     cout <<" bcr="<<bcr<<": Nr tracks "<<pileupTracks_[i].size()<<",";
// //     if (level>=3) {
// //       cout<<endl;
// //       for (unsigned int j=0;j<pileupTracks_[i].size();++j) 
// // 	cout<<" track "<<j<<" has vertex pointer "<< (pileupTracks_[i])[j].vertIndex()<<" and genpartindex "<<(pileupTracks_[i])[j].genpartIndex()<<endl;
// //     }
//   }
//   cout<<endl;
//   cout <<"\n Verticess "<<endl;
//   for (unsigned int i=0;i<pileupVertices_.size();++i) {
//     int bcr=firstCrossing_+i;
//     cout <<" bcr="<<bcr<<", Nr vtces="<<pileupVertices_[i].size();
//     // 	  if (level>=3) {
//     // 	  for (unsigned int j=0;j<pileupVertices_[i].size();++j) 
//     // 	    cout<<" vertex "<<j<<" has track pointer "<< (pileupVertices_[i])[j].parentIndex()<<endl;
//     // 	  }
}


std::ostream &operator<<(std::ostream& o, const CrossingFrame &cf)
{
  std::pair<int,int> range=cf.getBunchRange();
  o <<"\nCrossingFrame for "<<cf.getEventID()<<",  bunchrange = "<<range.first<<","<<range.second
	   <<", bunchSpace "<<cf.getBunchSpace();

  return o;
}


