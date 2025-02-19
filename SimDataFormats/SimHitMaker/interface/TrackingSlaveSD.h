#ifndef TrackingSlaveSD_h
#define TrackingSlaveSD_h

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <string>
#include <vector>
class SimTrackManager;

class TrackingSlaveSD
{
public:    
  typedef std::vector<PSimHit> Collection;
  typedef Collection::const_iterator const_iterator;
    TrackingSlaveSD(std::string);
    virtual ~TrackingSlaveSD();
    virtual void Initialize();
    //    virtual void renumbering(const SimTrackManager*); 
    virtual bool processHits(const PSimHit&);
    virtual bool format();
    std::string name() const { return name_; } 
    std::vector<PSimHit>& hits(){return hits_;}
    std::string type(){return "tk";}
    virtual const_iterator begin() { return hits_.begin();}
    virtual const_iterator end()   { return hits_.end();}

protected: 
    std::vector<PSimHit> hits_;
    void setTrackId(PSimHit & hit, unsigned int k);
private:
    std::string name_;
};

#endif 





