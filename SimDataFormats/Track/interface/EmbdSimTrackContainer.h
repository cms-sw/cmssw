#ifndef EmbdSimTrackContainer_H
#define EmbdSimTrackContainer_H

#include "SimDataFormats/Track/interface/EmbdSimTrack.h"

#include <vector>
#include <string>
 
namespace edm 
{
  class EmbdSimTrackContainer 
    {
    public:
      typedef std::vector<EmbdSimTrack> SimTrackContainer;
      void insertTrack( const EmbdSimTrack & t) { data.push_back(t); }
      void clear() { data.clear(); }
      unsigned int size () const {return data.size();}
      EmbdSimTrack operator[] (int i) const {return data[i]; }
 
      SimTrackContainer::const_iterator begin () const {return data.begin();}
      SimTrackContainer::const_iterator end () const  {return data.end();}

    private:
      SimTrackContainer data;
    };
} 
 

#endif
