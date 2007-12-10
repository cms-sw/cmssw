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

  CrossingFramePlaybackInfo()  {
   for (unsigned int i=0;i<maxNbSources;++i) {
     pileupFileNr_[i]=-1;
     idFirstPileup_[i]=edm::EventID(0,0);
   }
;}

  ~CrossingFramePlaybackInfo() {;}

  // getters 
  int getStartFileNr(const unsigned int s) const {return pileupFileNr_[s];}
  edm::EventID getStartEventId(const unsigned int s) const {return idFirstPileup_[s];}

  // setters 
  //FIXME: max nr sources, test on max nrsources
  void setStartFileNr(const  unsigned int nr, const unsigned int s) {pileupFileNr_[s]=nr;}
  void setStartEventId( const edm::EventID &id, const unsigned int s) {idFirstPileup_[s]=id;}
  void setEventStartInfo(const edm::EventID &id, int fileNr, const unsigned int s)
    {idFirstPileup_[s]=id;pileupFileNr_[s]=fileNr;
}


private:

  // for playback option
  static const unsigned int maxNbSources =4 ;  //FIXME: take from CrossingFrame
  edm::EventID idFirstPileup_[maxNbSources];   // EventId fof the first pileup event used for this signal event
  int pileupFileNr_[maxNbSources];    // ordinal number of the pileup file this event was in

};


#endif 
