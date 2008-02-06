#ifndef RPCDigitizer_RPCSim_h
#define RPCDigitizer_RPCSim_h

/** \class RPCSim
 *   Base Class for the RPC strip response simulation
 *  
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EventSetup.h>

#include <map>
#include <set>

#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"

class RPCRoll;
class RPCGeometry;
class RPCSimSetUp;
class PSimHit;

class RPCSim
{
 public:

  typedef edm::DetSet<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSet<RPCDigiSimLink> RPCDigiSimLinks;

  virtual ~RPCSim(){};

  virtual void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits)=0;

  virtual void simulateNoise(const RPCRoll* roll)=0;

  virtual void fillDigis(int rollDetId, RPCDigiCollection& digis);

  void setRPCSimSetUp(RPCSimSetUp* setup){theSimSetUp = setup;}

  RPCSimSetUp* getRPCSimSetUp(){ return theSimSetUp; }

  const DigiSimLinks & digiSimLinks() const {return theDigiSimLinks;}
  const RPCDigiSimLinks & rpcDigiSimLinks() const {return theRpcDigiSimLinks;}

 protected:
  RPCSim(const edm::ParameterSet& config);
  virtual void init()=0;

 protected:
  std::set< std::pair<int,int> > strips;

  //--------NEW---------------------

  /// creates links from Digi to SimTrack
  /// disabled for now
    virtual void addLinks(unsigned int strip,int bx);

  // keeps track of which hits contribute to which channels
    typedef std::multimap<std::pair<unsigned int,int>,const PSimHit*,std::less<std::pair<unsigned int, int> > >  DetectorHitMap;

  DetectorHitMap theDetectorHitMap;
  DigiSimLinks theDigiSimLinks;
  RPCDigiSimLinks theRpcDigiSimLinks;

  //--------------------------------

 protected:
  RPCSimSetUp* theSimSetUp;
};
#endif
