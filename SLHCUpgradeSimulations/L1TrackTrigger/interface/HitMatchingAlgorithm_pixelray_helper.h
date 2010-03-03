#ifndef HIT_MATCHING_ALGORITHM_pixelray_helper_H
#define HIT_MATCHING_ALGORITHM_pixelray_helper_H

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"

#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

namespace cmsUpgrades {

template<typename T>
MeasurementPoint getMpAverage(std::vector<T> hits)
{
    typename std::vector<T>::const_iterator hitsIter;
    
    float rowSum = 0;
    float colSum = 0;
    
    for (hitsIter = hits.begin(); hitsIter < hits.end(); hitsIter++)
    {
        rowSum += (*hitsIter)->row();
        colSum += (*hitsIter)->column();
    }
    
    return MeasurementPoint(rowSum / hits.size(), colSum / hits.size());
}

template<typename T>
std::pair<double,double> *
getPixelRayEndpoints(const LocalStub<T> & aLocalStub,
                     const StackedTrackerGeometry * stackedTracker,
                     double scalingFactor)
{
    // Get the coordinates of the boundaries of the inner and outer pixels.
    // Code adapted from LocalStub::averagePosition

    const GeomDetUnit* innerDet = stackedTracker->idToDetUnit(aLocalStub.Id(), 0);
    const GeomDetUnit* outerDet = stackedTracker->idToDetUnit(aLocalStub.Id(), 1);
    
    MeasurementPoint innerAvg = getMpAverage(aLocalStub.hit(0));
    MeasurementPoint outerAvg = getMpAverage(aLocalStub.hit(1));
    
    Measurement2DVector pixRtlOffset(0, 1);
    
    GlobalPoint innerPixLeft = innerDet->toGlobal(innerDet->topology().localPosition(
        innerAvg + pixRtlOffset));
    GlobalPoint innerPixRight = innerDet->toGlobal(innerDet->topology().localPosition(
        innerAvg));
    
    GlobalPoint outerPixLeft = outerDet->toGlobal(outerDet->topology().localPosition(
        outerAvg + pixRtlOffset));
    GlobalPoint outerPixRight = outerDet->toGlobal(outerDet->topology().localPosition(
        outerAvg));
    
    if (outerPixLeft.perp() < innerPixLeft.perp()) {
        // Do a grand swap
        GlobalPoint temp = innerPixLeft;
        innerPixLeft = outerPixLeft;
        outerPixLeft = temp;
        
        temp = innerPixRight;
        innerPixRight = outerPixRight;
        outerPixRight = temp;
    }
    
    // Left and right pixel boundaries differ only in z, have same r and \phi
    double outerPointRadius = outerPixLeft.perp();
    double innerPointRadius = innerPixLeft.perp();
    double outerPointPhi = outerPixLeft.phi();
    double innerPointPhi = innerPixLeft.phi();
    
    assert(outerPointRadius > innerPointRadius);
    assert(innerPixLeft.z() < innerPixRight.z());
    assert(outerPixLeft.z() < outerPixRight.z());
    
    // Check for seed compatibility given a pt cut
    // Threshold computed from radial location of hits
    double deltaPhiThreshold = (outerPointRadius - innerPointRadius) *
        scalingFactor;
    
    // Delta phi is computed from phi locations of hits, with
    // special case for delta phi arc passing through branch cut
    double deltaPhi = outerPointPhi - innerPointPhi;
    if (deltaPhi < 0) deltaPhi = -deltaPhi;
    
    if (deltaPhi > KGMS_PI) deltaPhi = 2 * KGMS_PI - deltaPhi;

    if (deltaPhi < deltaPhiThreshold)
    {
        double outerPixLeftZ  = outerPixLeft.z();
        double outerPixRightZ = outerPixRight.z();
        double innerPixLeftZ  = innerPixLeft.z();
        double innerPixRightZ = innerPixRight.z();
        
        // The projection factor relates distances on the detector
        // to distances on the beam.
        double projectFactor = outerPointRadius /
            (outerPointRadius - innerPointRadius);
        
        // The location of the projected pixel ray boundary
        double rightPixelRayZ = outerPixLeftZ +
            (innerPixRightZ - outerPixLeftZ) * projectFactor;
        double leftPixelRayZ = outerPixRightZ +
            (innerPixLeftZ - outerPixRightZ) * projectFactor;
        
        std::pair<double,double> * rayPair =
            new std::pair<double,double>(leftPixelRayZ, rightPixelRayZ);

        return rayPair;
    }
    
    return 0;
}

}

#endif

