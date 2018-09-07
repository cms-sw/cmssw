#ifndef SimG4CMS_Muon_MuonSlaveSD_h
#define SimG4CMS_Muon_MuonSlaveSD_h

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
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include <string>

class EndOfEvent;
class EventAction;
class MuonSubDetector;
class SimTrackManager;

class MuonSlaveSD : 
public TrackingSlaveSD
{
public: 
  typedef std::vector<PSimHit> Collection;
  typedef Collection::const_iterator const_iterator;
  MuonSlaveSD(MuonSubDetector*,const SimTrackManager*);
  ~MuonSlaveSD() override;
  virtual void clearHits();
  bool format() override;
  const_iterator begin() override { return hits_.begin();}
  const_iterator end() override   { return hits_.end();}

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
