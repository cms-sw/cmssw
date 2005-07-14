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
	void insertTrack(EmbdSimTrack & t) { data.push_back(t); }
	void clear() { data.clear(); }
    private:
	SimTrackContainer data;
    };
} 
 

#endif
