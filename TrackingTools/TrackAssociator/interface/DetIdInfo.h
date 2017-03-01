#ifndef TrackAssociator_DetIdInfo_h
#define TrackAssociator_DetIdInfo_h

/**
 * 
 *  Description:
 *    Helper class to get human readable informationa about given DetId
 * 
 */

#include "DataFormats/DetId/interface/DetId.h"
#include <set>
#include <vector>

class TrackerTopology;

class DetIdInfo {
 public:
  static std::string info( const DetId&, const TrackerTopology *tTopo );
  static std::string info( const std::set<DetId>&, const TrackerTopology *tTopo );
  static std::string info( const std::vector<DetId>&, const TrackerTopology *tTopo );
};
#endif
