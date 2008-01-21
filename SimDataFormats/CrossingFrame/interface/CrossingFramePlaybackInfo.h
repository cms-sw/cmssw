#ifndef CROSSING_FRAME_PLAYBACKINFO_H
#define CROSSING_FRAME_PLAYBACKINFO_H

/** \class CrossingFramePlaybackInfo
 *
 * CrossingFramePlaybackInfo is written by the Sim Mixing Module
 * it contains information to allow a 'playback' of the MixingModule
 * i.e to find again, on an event/event basis, exactly the same events to superpose
 *
 * \author Ursula Berthon, Claude Charlot,  LLR Palaiseau
 *
 * \version   1st Version Nov 2007
 *
 ************************************************************/


#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class CrossingFramePlaybackInfo 
{ 

 public:
  // con- and destructors

  CrossingFramePlaybackInfo() {;}
  CrossingFramePlaybackInfo(int minBunch, int maxBunch);

  ~CrossingFramePlaybackInfo() {;}

  // getters 
  int getStartFileNr(const unsigned int s,const int bcr) const {return (pileupFileNr_[s])[bcr-minBunch_];}
  edm::EventID getStartEventId(const unsigned int s,const int bcr) const {return (idFirstPileup_[s])[bcr-minBunch_];}
  int getNrEvents(const unsigned int s,const int bcr) const {return (nrEvents_[s])[bcr-minBunch_];}

  void getEventStartInfo(std::vector<edm::EventID> &ids, std::vector<int> &  fileNrs, std::vector<unsigned int> &nrEvents, const unsigned int s) const {
    ids=idFirstPileup_[s];
    fileNrs= pileupFileNr_[s];
    nrEvents=nrEvents_[s];
  }

  // setters 
  //FIXME: max nr sources, test on max nrsources
  void setStartFileNr(const  unsigned int nr, const unsigned int s,const int bcr) {pileupFileNr_[s][bcr-minBunch_]=nr;}
  void setStartEventId( const edm::EventID &id, const unsigned int s, const int bcr) {idFirstPileup_[s][bcr-minBunch_]=id;}
  void setNrEvents(const  unsigned int nr, const unsigned int s, const int bcr) {nrEvents_[s][bcr-minBunch_]=nr;}
/*   void setEventStartInfo(const edm::EventID &id, int fileNr, int nrEvents, const unsigned int s, const int bcr) */
/*     {idFirstPileup_[s][bcr-minBunch_]=id; */
/*      pileupFileNr_[s][bcr-minBunch_]=fileNr; */
/*      nrEvents_[s][bcr-minBunch_]=nrEvents; */
/* } */
  void setEventStartInfo(std::vector<edm::EventID> &id, std::vector<int>& fileNr, std::vector<unsigned int>& nrEvents, const unsigned int s);

private:

  // we need the same info for each bunchcrossing
  static const unsigned int maxNbSources =4 ;  //FIXME: take from CrossingFrame
std::vector<edm::EventID> idFirstPileup_[maxNbSources];   // EventId fof the first pileup event used for this signal event
std::vector<int> pileupFileNr_[maxNbSources];             // ordinal number of the pileup file this event was in
std::vector<unsigned int> nrEvents_[maxNbSources];

int nBcrossings_;
int minBunch_;

};


#endif 
