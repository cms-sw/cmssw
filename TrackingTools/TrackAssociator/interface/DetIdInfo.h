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

class DetIdInfo {
 public:
   static std::string info( const DetId& );
   static std::string info( const std::set<DetId>& );
   static std::string info( const std::vector<DetId>& );
};
#endif
