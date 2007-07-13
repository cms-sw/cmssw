#ifndef TkDetLayers_DetGroupMerger_h
#define TkDetLayers_DetGroupMerger_h

#include "TrackingTools/DetLayers/interface/DetGroup.h"

class DetGroupMerger {
public:

  void orderAndMergeTwoLevels( const std::vector<DetGroup>& one,  
						const std::vector<DetGroup>& two,
						std::vector<DetGroup>& result,
						int firstIndex, 
						int firstCrossed) const;

  void mergeTwoLevels( const std::vector<DetGroup>& one,  
					const std::vector<DetGroup>& two,
					std::vector<DetGroup>& result) const;

  void addSameLevel( const std::vector<DetGroup>& gvec, std::vector<DetGroup>& result) const;

  void doubleIndexSize( std::vector<DetGroup>& vec) const;

  void incrementAndDoubleSize( std::vector<DetGroup>& vec) const;

};

#endif
