#ifndef CROSSING_FRAME_PLAYBACKINFOEXTENDED_H
#define CROSSING_FRAME_PLAYBACKINFOEXTENDED_H

/** \class CrossingFramePlaybackInfoExtended
 *
 * CrossingFramePlaybackInfoExtended is written by the Sim Mixing Module
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
#include <vector>
#include <utility>

#include <iostream>

class CrossingFramePlaybackInfoExtended 
{ 

 public:
  // con- and destructors

  CrossingFramePlaybackInfoExtended() {;}
  CrossingFramePlaybackInfoExtended(int minBunch, int maxBunch, unsigned int maxNbSources);

  ~CrossingFramePlaybackInfoExtended() {;}

  // getters
  std::vector<edm::EventID> getStartEventId(const unsigned int s,const int bcr) const {return (idFirstPileup_[s])[bcr-minBunch_];}
  
  void getEventStartInfo(std::vector<std::vector<edm::EventID> > &ids, const unsigned int s) const {
    ids=idFirstPileup_[s];
  }
  
   
  // setters 
  //FIXME: max nr sources, test on max nrsources
  void setStartEventId( const std::vector<edm::EventID> &id, const unsigned int s, const int bcr, const int start) {
    std::vector<edm::EventID> newVec;
    std::vector<edm::EventID>::const_iterator idStart = id.begin()+start;
    newVec.insert(newVec.begin(), idStart, id.end());
    idFirstPileup_[s][bcr-minBunch_]=newVec;
  }
  void setEventStartInfo(std::vector<std::vector<edm::EventID> > &id, const unsigned int s);

 private:

  // we need the same info for each bunchcrossing
  unsigned int maxNbSources_;
  
  std::vector<std::vector<std::vector<edm::EventID> > > idFirstPileup_;
 
  int nBcrossings_;
  int minBunch_;
};


#endif 
