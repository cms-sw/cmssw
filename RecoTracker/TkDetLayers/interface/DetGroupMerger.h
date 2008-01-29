#ifndef TkDetLayers_DetGroupMerger_h
#define TkDetLayers_DetGroupMerger_h

#include "TrackingTools/DetLayers/interface/DetGroup.h"

class DetGroupMerger {
public:

  std::vector<DetGroup> orderAndMergeTwoLevels( const std::vector<DetGroup>& one,  
						const std::vector<DetGroup>& two,
						int firstIndex, 
						int firstCrossed) const;

  std::vector<DetGroup> mergeTwoLevels( const std::vector<DetGroup>& one,  
					const std::vector<DetGroup>& two) const;

  void addSameLevel( const std::vector<DetGroup>& gvec, std::vector<DetGroup>& result) const;

  void doubleIndexSize( std::vector<DetGroup>& vec) const;

  void incrementAndDoubleSize( std::vector<DetGroup>& vec) const;

};

#endif
