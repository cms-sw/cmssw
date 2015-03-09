#ifndef CROSSING_FRAME_PLAYBACKINFONEW_H
#define CROSSING_FRAME_PLAYBACKINFONEW_H

/** \class CrossingFramePlaybackInfoNew
 *
 * CrossingFramePlaybackInfoNew is written by the Sim Mixing Module
 * it contains information to allow a 'playback' of the MixingModule
 * i.e to find again, on an event/event basis, exactly the same events to superpose
 *
 * \author Bill Tanenbaum
 *
 * \version   1st Version Feb 2015
 *
 ************************************************************/


#include "DataFormats/Common/interface/SecondaryEventIDAndFileInfo.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iterator>
#include <vector>
#include <utility>

class CrossingFramePlaybackInfoNew 
{ 

 public:
  // con- and destructors

  CrossingFramePlaybackInfoNew() {}
  CrossingFramePlaybackInfoNew(int minBunch, int maxBunch, unsigned int maxNbSources);

  ~CrossingFramePlaybackInfoNew() {}

  typedef std::vector<edm::SecondaryEventIDAndFileInfo>::iterator iterator;
  typedef std::pair<iterator, iterator> range;

  // setter
  void setInfo(std::vector<edm::SecondaryEventIDAndFileInfo>& eventInfo, std::vector<size_t>& sizes) {
    sizes_.swap(sizes);
    eventInfo_.swap(eventInfo);
  }
 
  // getters
  std::vector<edm::SecondaryEventIDAndFileInfo>::const_iterator getEventId(size_t offset) const {
    std::vector<edm::SecondaryEventIDAndFileInfo>::const_iterator iter = eventInfo_.begin();
    std::advance(iter, offset);
    return iter;
  }

  size_t getNumberOfEvents(int bunchIdx, size_t sourceNumber) const {
     return sizes_[((bunchIdx - minBunch_) * maxNbSources_) + sourceNumber];
  }

 //private:

  // we need the same info for each bunchcrossing
  unsigned int maxNbSources_;
  int nBcrossings_;
  std::vector<size_t> sizes_;
  std::vector<edm::SecondaryEventIDAndFileInfo> eventInfo_;
  int minBunch_;
};


#endif 
