#ifndef RecoPixelVertexing_FindPeakFastPV_h
#define RecoPixelVertexing_FindPeakFastPV_h
/** \class FindPeakFastPV FindPeakFastPV.h RecoTracker/PixelVertexFinding/FindPeakFastPV.h 
 * Given *zProjections* and *zWeights* find the peak of width *m_zClusterWidth*. 
 * Use only values with *zWeights*>*m_weightCut*. 
 * Look near *oldVertex* within *m_zClusterSearchArea*.
 *
 *  \author Silvio Donato (SNS)
 */

#include <vector>

inline float FindPeakFastPV(const std::vector<float> &zProjections,
                            const std::vector<float> &zWeights,
                            const float oldVertex,
                            const float m_zClusterWidth,
                            const float m_zClusterSearchArea,
                            const float m_weightCut) {
  float centerWMax = oldVertex;
  if (m_zClusterWidth > 0 && m_zClusterSearchArea > 0) {
    std::vector<float>::const_iterator itCenter = zProjections.begin();
    std::vector<float>::const_iterator itLeftSide = zProjections.begin();
    std::vector<float>::const_iterator itRightSide = zProjections.begin();
    const float zClusterWidth = m_zClusterWidth * 0.5;  //take half zCluster width
    const float zLowerBound = oldVertex - zClusterWidth - m_zClusterSearchArea;
    const float zUpperBound = oldVertex + zClusterWidth + m_zClusterSearchArea;
    float maxW = 0;
    for (; itCenter != zProjections.end(); itCenter++) {
      //loop only on the zProjections within oldVertex +/- (zClusterWidth + m_zClusterSearchArea)
      if ((*itCenter < zLowerBound) || (*itCenter > zUpperBound))
        continue;

      while (itLeftSide != zProjections.end() && (*itCenter - *itLeftSide) > zClusterWidth)
        itLeftSide++;
      while (itRightSide != zProjections.end() && (*itRightSide - *itCenter) < zClusterWidth)
        itRightSide++;
      float nWeighted = 0;
      float centerW = 0;

      for (std::vector<float>::const_iterator ii = itLeftSide; ii != itRightSide; ii++) {
        //loop inside the peak and calculate its weight
        if (zWeights[ii - zProjections.begin()] < m_weightCut)
          continue;
        nWeighted += zWeights[ii - zProjections.begin()];
        centerW += (*ii) * zWeights[ii - zProjections.begin()];
      }
      centerW /= nWeighted;  //calculate the weighted peak center
      if (nWeighted > maxW) {
        maxW = nWeighted;
        centerWMax = centerW;
      }
    }
  }
  return centerWMax;
}

#endif
