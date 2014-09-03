#ifndef MuonSlaveSD_h
#define MuonSlaveSD_h

/** \class MuonSlaveSD
 *
 * a copy of the TrackingSlaveSD extended by
 * muon hit formatting; interface to the database 
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class EndOfEvent;
class EventAction;

#include <string>

class MuonSubDetector;
class SimTrackManager;
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

class MuonSlaveSD : 
public TrackingSlaveSD
{
public: 
  typedef std::vector<PSimHit> Collection;
  typedef Collection::const_iterator const_iterator;
  MuonSlaveSD(MuonSubDetector*,const SimTrackManager*);
  virtual ~MuonSlaveSD();
  virtual void clearHits();
  virtual bool format();
  virtual const_iterator begin() { return hits_.begin();}
  virtual const_iterator end()   { return hits_.end();}

protected: 
  Collection 	     hits_;

private:
  MuonSubDetector* detector;

  const SimTrackManager* m_trackManager;

};

class FormatBarrelHits {
 public:
  bool operator() (const PSimHit & a, const PSimHit & b);
  int sortId (const PSimHit & a)  const;
};

class FormatEndcapHits {
 public:
  bool operator() (const PSimHit & a, const PSimHit & b);
  int sortId (const PSimHit & a)  const;
};

class FormatRpcHits {
 public:
  bool operator() (const PSimHit & a, const PSimHit & b);
  int sortId (const PSimHit & a)  const;
};

class FormatGemHits {
 public:
  bool operator() (const PSimHit & a, const PSimHit & b);
  int sortId (const PSimHit & a)  const;
};

class FormatMe0Hits {
 public:
  bool operator() (const PSimHit & a, const PSimHit & b);
  int sortId (const PSimHit & a)  const;
};

#endif // MuonSlaveSD_h
