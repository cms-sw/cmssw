#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
using namespace std;
using namespace edm;

    CrossingFrame::CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> trackersubdetectors,std::vector<std::string> calosubdetectors): BCrossingFrame(minb,bunchsp) {

    // create and insert vectors into the map
    // for tracker
    for(std::vector<std::string >::const_iterator it = trackersubdetectors.begin(); it != trackersubdetectors.end(); ++it) {  
      vector<PSimHitContainer> myvec(-minb+maxb+1);
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


  void CrossingFrame::addPileupSimHits(const int bcr, const std::string subdet, const PSimHitContainer *simhits) { 
    for (unsigned int i=0;i<simhits->size();++i) 
      (pileupSimHits_[subdet])[bcr-firstCrossing_].insertHit((*simhits)[i]);
  }

  void CrossingFrame::addPileupCaloHits(const int bcr, const std::string subdet, const PCaloHitContainer *calohits) { 
    for (unsigned int i=0;i<calohits->size();++i) 
      (pileupCaloHits_[subdet])[bcr-firstCrossing_].insertHit((*calohits)[i]);
  }

  void CrossingFrame::addPileupTracks(const int bcr, const EmbdSimTrackContainer *simtracks) { 
    for (unsigned int i=0;i<simtracks->size();++i) 
    pileupTracks_[bcr-firstCrossing_].insertTrack((*simtracks)[i]);
  }

  void CrossingFrame::addPileupVertices(const int bcr, const EmbdSimVertexContainer *simvertices) { 
    for (unsigned int i=0;i<simvertices->size();++i) 
    pileupVertices_[bcr-firstCrossing_].insertVertex((*simvertices)[i]);
  }

  void CrossingFrame::print(int level) const {
    BCrossingFrame::print(level);
    //
    // print for lowest level (signals)
    //
    cout<<"\nSignals:"<<endl;
    for(map<string,PSimHitContainer>::const_iterator it = signalSimHits_.begin(); it != signalSimHits_.end(); ++it) 
      cout<< " subdetector "<<(*it).first <<" signal size "<<(*it).second.size()<<endl;

    for(map<string,PCaloHitContainer>::const_iterator it = signalCaloHits_.begin(); it != signalCaloHits_.end(); ++it) 
      cout<< " subdetector "<<(*it).first <<" signal size"<<(*it).second.size()<<endl;

    cout <<" Number of tracks in signal "<< signalTracks_.size()<<endl;
    cout <<" Number of vertices in signal "<< signalVertices_.size()<<endl;

    //
    // print for next higher level (pileups)
    //
    if (level<1) return;
    cout<<"\nPilups:"<<endl;
    map<string,vector<PSimHitContainer> >::const_iterator itsim;
    for(itsim = pileupSimHits_.begin(); itsim != pileupSimHits_.end(); ++itsim){ 
      cout<< " Pileup for subdetector "<<(*itsim).first <<endl;
      for (unsigned int i=0;i<(*itsim).second.size();++i) {
	int bcr=firstCrossing_+i;
        cout <<" Bunchcrossing  "<<bcr<<", Simhit pileup size "<<(*itsim).second[i].size()<<endl;
      }
    }

    map<string,vector<PCaloHitContainer> >::const_iterator it;
    for(it = pileupCaloHits_.begin(); it != pileupCaloHits_.end(); ++it){ 
      cout<< " Pileup for subdetector "<<(*it).first <<endl;
      for (unsigned int i=0;i<(*it).second.size();++i) {
	int bcr=firstCrossing_+i;
        cout <<" Bunchcrossing  "<<bcr<<", Calohit pileup size "<<(*it).second[i].size()<<endl;
      }
    }

    for (unsigned int i=0;i<pileupTracks_.size();++i) {
	int bcr=firstCrossing_+i;
        cout <<" Bunchcrossing  "<<bcr<<", Nr  pileup tracks "<<pileupTracks_[i].size();
	cout<<", Nr  pileup vertices "<<pileupVertices_[i].size( )<<endl;
    }
   
    //
    // print for next higher level
    //
    if (level<2) return;
  }

