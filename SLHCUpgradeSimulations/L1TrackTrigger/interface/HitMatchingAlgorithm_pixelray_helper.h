/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGORITHM_pixelray_helper_H
#define HIT_MATCHING_ALGORITHM_pixelray_helper_H

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

  using namespace std;

  /// Get pixel ray end points
  template< typename T >
  std::pair< double, double > * getPixelRayEndpoints( const L1TkStub< T > & aL1TkStub,
                                                      const StackedTrackerGeometry * stackedTracker,
                                                      double scalingFactor )
  {
    /// Get the coordinates of the boundaries of the inner and outer pixels.
    /// Code adapted from Cluster::averagePosition
    const GeomDetUnit* innerDet = stackedTracker->idToDetUnit(aL1TkStub.getDetId(), 0);
    const GeomDetUnit* outerDet = stackedTracker->idToDetUnit(aL1TkStub.getDetId(), 1);

    MeasurementPoint innerAvg = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
    MeasurementPoint outerAvg = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();

    StackedTrackerDetId innerDetId( aL1TkStub.getClusterPtr(0)->getDetId() );
    StackedTrackerDetId outerDetId( aL1TkStub.getClusterPtr(1)->getDetId() );
    unsigned int innerStackMember = aL1TkStub.getClusterPtr(0)->getStackMember();
    unsigned int outerStackMember = aL1TkStub.getClusterPtr(1)->getStackMember();
    unsigned int innerStack = innerDetId.iLayer();
    unsigned int outerStack = outerDetId.iLayer();
    unsigned int innerLadderPhi = innerDetId.iPhi();
    unsigned int outerLadderPhi = outerDetId.iPhi();
    unsigned int innerLadderZ = innerDetId.iZ();
    unsigned int outerLadderZ = outerDetId.iZ();

    Measurement2DVector pixRtlOffset(0, 1);

    /// Find leftmost and rightmost pixels of projection
    GlobalPoint innerPixLeft  = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg + pixRtlOffset ));
    GlobalPoint innerPixRight = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg ));
    GlobalPoint outerPixLeft  = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg + pixRtlOffset ));
    GlobalPoint outerPixRight = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg ));
    GlobalPoint oldInnerPixLeft  = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg + pixRtlOffset ));
    GlobalPoint oldInnerPixRight = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg ));
    GlobalPoint oldOuterPixLeft  = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg + pixRtlOffset ));
    GlobalPoint oldOuterPixRight = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg ));

    bool swap = false;
    
    /// Cross check and swap left/right
    if ( outerPixLeft.perp() < innerPixLeft.perp() )
    {
      GlobalPoint temp = innerPixLeft;
      innerPixLeft = outerPixLeft;
      outerPixLeft = temp;
     
      temp = innerPixRight;
      innerPixRight = outerPixRight;
      outerPixRight = temp;     
      swap = true;
    }
    
    /// Get useful quantities
    /// Left and right pixel boundaries differ only in z, have same r and \phi
    double outerPointRadius = outerPixLeft.perp();
    double innerPointRadius = innerPixLeft.perp();

    double outerPointPhi = outerPixLeft.phi();
    double innerPointPhi = innerPixLeft.phi();
    double outerPointEta = outerPixLeft.eta();
    double innerPointEta = innerPixLeft.eta();

    double outerPixLeftX = outerPixLeft.x();
    double innerPixLeftX = innerPixLeft.x();
    double outerPixLeftY = outerPixLeft.y();
    double innerPixLeftY = innerPixLeft.y();

    double outerPixRightX = outerPixRight.x();
    double innerPixRightX = innerPixRight.x();
    double outerPixRightY = outerPixRight.y();
    double innerPixRightY = innerPixRight.y();

    double oldOuterPointRadius = oldOuterPixLeft.perp();
    double oldInnerPointRadius = oldInnerPixLeft.perp();
    double oldOuterPointPhi = oldOuterPixLeft.phi();
    double oldInnerPointPhi = oldInnerPixLeft.phi();
    double oldOuterPointEta = oldOuterPixLeft.eta();
    double oldInnerPointEta = oldInnerPixLeft.eta();

    if (outerPointRadius <= innerPointRadius || innerPixLeft.z() >= innerPixRight.z() || outerPixLeft.z() >= outerPixRight.z() )
    {
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", VALUE BEFORE THE SWAP " << endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", oldOuterPointRadius "<< oldOuterPointRadius << ", oldInnerPointRadius " << oldInnerPointRadius << endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", oldOuterPointPhi "<< oldOuterPointPhi << ", oldInnerPointPhi " << oldInnerPointPhi << endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", oldOuterPointEta "<< oldOuterPointEta << ", oldInnerPointEta " << oldInnerPointEta << endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", oldInnerPixLeft.z() "<< oldInnerPixLeft.z() << ", oldInnerPixRight.z() " << oldInnerPixRight.z() << endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", oldOuterPixLeft.z() "<< oldOuterPixLeft.z() << ", oldOuterPixRight.z() " << oldOuterPixRight.z() << endl;
      if (swap) cout << cout.precision(10) <<   endl;
      if (swap) cout << cout.precision(10) <<  __LINE__ << ", VALUE AFTER THE SWAP " << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPixLeftX " << outerPixLeftX  << ", innerPixLeftX "  << innerPixLeftX  << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPixLeftY " << outerPixLeftY  << ", innerPixLeftY "  << innerPixLeftY  << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPixRightX "<< outerPixRightX << ", innerPixRightX " << innerPixRightX << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPixRightY "<< outerPixRightY << ", innerPixRightY " << innerPixRightY << endl;

      cout << cout.precision(10) <<  __LINE__ << ", outerPointRadius "<< outerPointRadius << ", innerPointRadius " << innerPointRadius << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPointPhi "<< outerPointPhi << ", innerPointPhi " << innerPointPhi << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPointEta "<< outerPointEta << ", innerPointEta " << innerPointEta << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerPixLeft.z() "<< innerPixLeft.z() << ", innerPixRight.z() " << innerPixRight.z() << endl;
      cout << cout.precision(10) <<  __LINE__ << ", outerPixLeft.z() "<< outerPixLeft.z() << ", outerPixRight.z() " << outerPixRight.z() << endl;
      cout << cout.precision(10) <<  endl;
      cout << cout.precision(10) <<  __LINE__ << ", CLUSTER VALUES " << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerAvg "<< innerAvg << ", outerAvg "<< outerAvg  << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerDetId "<< innerDetId << ", outerDetId " << outerDetId << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerStackMember "<< innerStackMember << ", outerStackMember " << outerStackMember << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerStack "<< innerStack << ", outerStack " << outerStack << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerLadderPhi " << innerLadderPhi << ", outerLadderPhi " << outerLadderPhi << endl;
      cout << cout.precision(10) <<  __LINE__ << ", innerLadderZ " << innerLadderZ << ", outerLadderZ " << outerLadderZ << endl;
    }

    assert(outerPointRadius >= innerPointRadius);
    assert(innerPixLeft.z() < innerPixRight.z());
    assert(outerPixLeft.z() < outerPixRight.z());
    
    /// Check for seed compatibility given a pt cut
    /// Threshold computed from radial location of hits
    double deltaRadius = outerPointRadius - innerPointRadius;
    double deltaPhiThreshold = deltaRadius * scalingFactor;  

    /// Calculate angular displacement from hit phi locations
    /// and renormalize it, if needed
    double deltaPhi = outerPointPhi - innerPointPhi;
    if ( deltaPhi < 0 ) deltaPhi = -deltaPhi;
    if ( deltaPhi > M_PI ) deltaPhi = 2*M_PI - deltaPhi;

    /// Apply selection based on Pt
    if ( deltaPhi < deltaPhiThreshold )
    {
      /// Check for backprojection to beamline
      double outerPixLeftZ  = outerPixLeft.z();
      double outerPixRightZ = outerPixRight.z();
      double innerPixLeftZ  = innerPixLeft.z();
      double innerPixRightZ = innerPixRight.z();
        
      /// The projection factor relates distances on the detector to distances on the beam
      /// The location of the projected pixel ray boundary is then found
      double projectFactor = outerPointRadius / deltaRadius;
      double rightPixelRayZ = outerPixLeftZ + (innerPixRightZ - outerPixLeftZ) * projectFactor;
      double leftPixelRayZ = outerPixRightZ + (innerPixLeftZ - outerPixRightZ) * projectFactor;

      /// Return where ray ends
      std::pair< double, double > * rayPair = new std::pair< double, double >( leftPixelRayZ, rightPixelRayZ );
      return rayPair;
    }

    return 0;
  }



#endif

