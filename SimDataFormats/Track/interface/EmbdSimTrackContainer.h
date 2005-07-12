#ifndef EmbdSimTrackContainer_H
#define EmbdSimTrackContainer_H

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimDataFormats/SimEvent/interface/EmbdSimTrack.h"

#include <vector>
#include <string>
 
namespace edm 
{
    class EmbdSimTrackContainer: public EDProduct 
    {
    public:
	typedef std::vector<EmbdSimTrack> SimTrackContainer;
	void insertTrack(EmbdSimTrack & t) { data.push_back(t); }
    private:
	SimTrackContainer data;
    };
} 
 

#endif
