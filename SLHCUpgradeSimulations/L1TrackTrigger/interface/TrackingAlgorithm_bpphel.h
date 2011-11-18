/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Sept                           ///
///                                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGORITHM_bpphel_H
#define TRACKING_ALGORITHM_bpphel_H

#include <memory>
#include <string>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <boost/shared_ptr.hpp>

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class TrackingAlgorithm_bpphel : public TrackingAlgorithm< T > {

    private:
      /// Data members
      /// Propagation of seed
      double mWindowSize;            /// Matching window probability size from 336 studies
      double mMagneticFieldStrength; /// B field used for propagation
      edm::ParameterSet & mParSet;   /// ParameterSet to handle the LUT
      /// Other stuff
      const cmsUpgrades::classInfo *mClassInfo;

    public:
      /// Constructor
      TrackingAlgorithm_bpphel( const cmsUpgrades::StackedTrackerGeometry *i ,
                                double aMagneticFieldStrength,
                                double aWindowSize,
                                edm::ParameterSet & aParSet ) :
                                TrackingAlgorithm< T >( i ),
                                mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ),
                                mMagneticFieldStrength( aMagneticFieldStrength ),
                                mWindowSize( aWindowSize ),
                                mParSet( aParSet ) {}
      /// Destructor
      ~TrackingAlgorithm_bpphel() {}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Seed propagation
      std::vector< cmsUpgrades::L1TkTrack< T > > PropagateSeed( edm::Ptr< cmsUpgrades::L1TkTracklet< T > > aSeed, std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > aBricks ) const;
      /// Make the PSimHit equivalent
      std::pair< double, PSimHit > MakeHit( const GeomDetUnit* dU, BaseParticlePropagator* tP, double curv ) const;

      /// Algorithm name
      std::string AlgorithmName() const { 
        return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
      }

  }; /// Close class



  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Seed propagation
  template<typename T>
  std::vector< cmsUpgrades::L1TkTrack< T > > TrackingAlgorithm_bpphel< T >::PropagateSeed( edm::Ptr< cmsUpgrades::L1TkTracklet< T > > aSeed,
                                                                                           std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > aBricks ) const {
    /// Prepare output
    std::vector< cmsUpgrades::L1TkTrack< T > > tempTrackColl;
    tempTrackColl.clear();

    /// Already present from Algoritm
    /// double mWindowSize (this is passed in ES_TrackingAlgorithm_bpphel via cfi file)
    /// double mMagneticFieldStrength rounded to 3.8 or 4.0

    /// Step 00
    /// Linear interpolation of Pt;
    /// get Pt from seed, and find
    /// corresponding lookup tables
    std::vector< double > aPtRef = mParSet.getParameter< std::vector< double > >("aptref");
    double seedPt = aSeed->getMomentum().perp();
    int whichTable = -1; /// Should be unsigned int, but "-1" is just
                         /// a dummy to be sure of correct initialization
    /// Loop over Pt reference values
    for ( unsigned int i = 0; i < aPtRef.size(); i++ ) {
      if ( seedPt > aPtRef.at(i) ) whichTable = (int)i;
    }
    /// Now whichTable contains the index of the largest
    /// aPtRef lower than seedPt. i.e. if seedPt = 25 and
    /// aPtRef = {3,10,20,30} whichOne = 2. ok?

    /// Start checking different cases (in particular
    /// to take care of out-of-range seedPt)
    /// Calculate where the seedPt is in the interval
    double scaleFactor = 0;
    if ( whichTable == (int)aPtRef.size()-1 ) {
      whichTable -= 1;
      scaleFactor = 1;
    }
    else if ( whichTable < 0) {
      whichTable = 0;
      scaleFactor = (seedPt - aPtRef.at(whichTable)) / (aPtRef.at(whichTable+1) - aPtRef.at(whichTable));
      /// Here we keep linear interpolation also below 3 GeV/c Pt, using the same slope as
      /// in the first interval between 3 and 5 GeV/c
    }
    else scaleFactor = (seedPt - aPtRef.at(whichTable)) / (aPtRef.at(whichTable+1) - aPtRef.at(whichTable));

    /// Build names to get windows from configuration file
    /// Only base of names
    std::ostringstream phiBiasName0;
    std::ostringstream phiWinName0;
    std::ostringstream zWinName0;
    int probLabel = (int)mWindowSize;
    phiBiasName0 << "a" << "phibias";
    phiWinName0 << "a" << probLabel << "phiwin";
    zWinName0 << "a" << probLabel << "zwin";

    /// Step 0; constraints
    /// from superlayer [5] to layer [10]
    double biasPhi[5][10];
    double windowsPhi[5][10];
    double windowsZ[5][10];

    /// Initialize
    for ( unsigned int v = 0; v < 10; v++ ) {
      for ( unsigned int m = 0; m < 5; m++) {
        biasPhi[m][v] = 0;
        windowsPhi[m][v] = -9999.9;
        windowsZ[m][v] = -9999.9;
      }
    }

    std::vector<double> aBiasPhiRef;
    std::vector<double> aWinPhiRef;
    std::vector<double> aWinZRef;

    /// Get the charge of the seed
    double trkQ = aSeed->getCharge();

    /// Load Tables with constraints
    for ( unsigned int v = 0; v < 10; v++ ) {
      for ( unsigned int m = 0; m < 5; m++ ) {
        /// Reject bad Seed DoubleStack/Target Stack combinations and keep default
        if ( m==0 && v<2 ) continue;
        if ( m==1 && (v==2 || v==3) ) continue;
        if ( m>1 && v>3)  continue;

        /// Here we have only good pairs
        /// Complete names
        std::ostringstream phiBiasName1;
        std::ostringstream phiWinName1;
        std::ostringstream zWinName1;
        phiBiasName1 << phiBiasName0.str().c_str() << m << v;
        phiWinName1  << phiWinName0.str().c_str()  << m << v;
        zWinName1    << zWinName0.str().c_str()    << m << v;

        /// Get the parameters from configuration file
        aBiasPhiRef = mParSet.getParameter< std::vector< double > >( phiBiasName1.str().c_str() );
        aWinPhiRef  = mParSet.getParameter< std::vector< double > >( phiWinName1.str().c_str() );
        aWinZRef    = mParSet.getParameter< std::vector< double > >( zWinName1.str().c_str() );

        /// Fill the matching window from Seed DoubleStack m to Target Stack v
        /// Include calculation of left/right bias due to clockwise/counterclockwise
        /// propagation of the charged track
        biasPhi[m][v]    = trkQ*aBiasPhiRef.at(whichTable) + scaleFactor*( aBiasPhiRef.at(whichTable+1) - aBiasPhiRef.at(whichTable) );
        windowsPhi[m][v] = aWinPhiRef.at(whichTable)       + scaleFactor*( aWinPhiRef.at(whichTable+1) - aWinPhiRef.at(whichTable) );
        windowsZ[m][v]   = aWinZRef.at(whichTable)         + scaleFactor*( aWinZRef.at(whichTable+1)-aWinZRef.at(whichTable) );
      }
    } /// End of double loop over Seed DoubleStack and Target Stack

    /// Get Seed Stubs
    std::set< std::pair< unsigned int , edm::Ptr< cmsUpgrades::L1TkStub< T > > > > theStubs = aSeed->getStubRefs();
    edm::Ptr< cmsUpgrades::L1TkStub< T > > innerStub = theStubs.begin()->second;
    edm::Ptr< cmsUpgrades::L1TkStub< T > > outerStub = theStubs.rbegin()->second;

    /// Get Stack Idx
    unsigned int doubleStack = aSeed->getDoubleStack();
    unsigned int innerStack = innerStub->getStack();
    unsigned int outerStack = outerStub->getStack();

    /// Get parameters at Vertex
    GlobalPoint  vertexPosition = aSeed->getVertex();
    GlobalVector vertexMomentum = aSeed->getMomentum();
    double vertexCurvature = -cmsUpgrades::KGMS_C * 1e-11 * mMagneticFieldStrength * trkQ / vertexMomentum.perp(); /// The 1e-11 is to adjust the units of measurement

    /// Cross-check for self-consistency
    if ( doubleStack != innerStack/2 ) std::cerr << "HOUSTON WE GOT A PROBLEM!" << std::endl;
    if ( innerStack/2 != (outerStack-1)/2 ) std::cerr << "HOUSTON WE GOT A PROBLEM!" << std::endl;

    /// Step 1: Build Propagators
    /// Find Tracklet Direction in transverse plane
    /// to put in propagators to enable propagation
    /// from Tracklet and not from Vertex
    /// NOTE: this is written but commented since new methods
    /// to get Tracklet momentum at Stubs is available in 42X
    /// However, the algorithm was validated only in 336, so
    /// you must be careful of these small differences for the
    /// present moment
    /*
    GlobalVector trkDirection = outerStub->getPosition() - innerStub->getPosition();
    /// Renormalize to 1
    trkDirection = trkDirection / trkDirection.mag();
    /// Propagate the seed Tracklet
    /// First step is to construct the propagator
    double mom[4] = {0,0,0,0};
    /// Pz = Pt / tan theta
    mom[3] = aSeed->getMomentum().z();
    mom[2] = aSeed->getMomentum().perp() * trkDirection.y()/trkDirection.perp();
    mom[1] = aSeed->getMomentum().perp() * trkDirection.x()/trkDirection.perp();
    mom[0] = sqrt( 0.1*0.1 + mom[1]*mom[1] + mom[2]*mom[2] + mom[3]*mom[3] );
    math::XYZTLorentzVector trkMom( mom[1], mom[2], mom[3], mom[0] );
    /// Energy component is due to the fact that both pions and muons have ~100 MeV mass
    double pos[4] = {0,0,0,0};
    pos[0] = 0.0;
    pos[1] = innerStub->getPosition().x() + outerStub->getPosition().x();
    pos[2] = innerStub->getPosition().y() + outerStub->getPosition().y();
    pos[3] = innerStub->getPosition().z() + outerStub->getPosition().z();
    math::XYZTLorentzVector trkPos( 0.5*pos[1], 0.5*pos[2], 0.5*pos[3], pos[0] );
    /// We can declare a BaseParticlePropagator without time component of VTX
    RawParticle trkRaw( trkMom , trkPos );
    trkRaw.setCharge( trkQ );
    */

    /// Here, do the same things using position of Stubs and
    /// momentum calculated at Stub position according to
    /// inwards/outwards propagation!
    GlobalVector innerStubMom = aSeed->getMomentum(0);
    GlobalVector outerStubMom = aSeed->getMomentum(1);
    GlobalPoint  innerStubPos = innerStub->getPosition();
    GlobalPoint  outerStubPos = outerStub->getPosition();
    math::XYZTLorentzVector innerTrkMom( innerStubMom.x(), innerStubMom.y(), innerStubMom.z(), sqrt( 0.1*0.1 + innerStubMom.mag2() ) );
    math::XYZTLorentzVector outerTrkMom( outerStubMom.x(), outerStubMom.y(), outerStubMom.z(), sqrt( 0.1*0.1 + outerStubMom.mag2() ) );
    math::XYZTLorentzVector innerTrkPos( innerStubPos.x(), innerStubPos.y(), innerStubPos.z(), 0.0 );
    math::XYZTLorentzVector outerTrkPos( outerStubPos.x(), outerStubPos.y(), outerStubPos.z(), 0.0 );

    /// Declare the RawParticle for the BaseParticlePropagator
    /// No time component of vertex, order of magnitude pion mass
    RawParticle innerTrkRaw( innerTrkMom, innerTrkPos );
    innerTrkRaw.setCharge( trkQ );
    RawParticle outerTrkRaw( outerTrkMom, outerTrkPos );
    outerTrkRaw.setCharge( trkQ );

    /// Define propagation limits
    /// Remember that short barrels count twice, however, as BaseParticlePropagator
    /// thinks only in terms of symmetric z bounds, we will work with a double
    /// propagation with an accept-forward/exclude-central anticoincidence
    double StackR[14] = { 32.0, 36.0,
                          48.0, 52.0,
                          64.3, 68.3, /// Short Barrel
                          80.3, 84.3, /// Short Barrel
                          98.5, 102.5,
                          64.3, 68.3,   /// Veto
                          80.3, 84.3 }; /// Veto
    double StackZ[14] = { 420.0, 420.0,
                          540.0, 540.0,
                          540.0, 540.0, /// Short Barrel
                          540.0, 540.0, /// Short Barrel
                          540.0, 540.0,
                          420.0, 420.0,    /// Veto
                          420.0, 420.0 };  /// Veto

    /// Build propagators
    /// Do it only for the correct DoubleStack/Stack pairs
    BaseParticlePropagator* trkProp[14];
 
    /// Take into account only Target Stacks
    /// as Seed DoubleStacks are already included in the
    /// loop over seeds
    for ( unsigned int k = 0; k < 10; k++ ) {
      /// Do not propagate to the same layers as the seed
      if ( k==innerStack || k==outerStack ) continue;
      else if ( k > outerStack ) {
        /// Forward propagation from outer Stub
        trkProp[k] = new BaseParticlePropagator( outerTrkRaw, StackR[k], StackZ[k]/2.0, mMagneticFieldStrength );
        trkProp[k]->propagate();
        if ( k>3 && k<8 ) {
          /// Veto in short barrel
          trkProp[k+6] = new BaseParticlePropagator( outerTrkRaw, StackR[k+6], StackZ[k+6]/2.0, mMagneticFieldStrength );
          trkProp[k+6]->propagate();
        }
      }
      else {
        /// Backward propagation from inner Stub
        trkProp[k] = new BaseParticlePropagator( innerTrkRaw, StackR[k], StackZ[k]/2.0, mMagneticFieldStrength );
        trkProp[k]->backPropagate();
        if ( k>3 && k<8 ) {
          /// Veto in short barrel
          trkProp[k+6] = new BaseParticlePropagator( innerTrkRaw, StackR[k+6], StackZ[k+6]/2.0, mMagneticFieldStrength );
          trkProp[k+6]->backPropagate();
        }
      }      
    } /// End of Propagator construction

    /// Step 2: Associate Stubs to seed
    /// Ready to match Stubs
    /// Store them in a Stack-wise way
    std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > tempStubs[10];
    tempStubs[innerStack].push_back( innerStub );
    tempStubs[outerStack].push_back( outerStub );

    /// Loop over Bricks
    for ( unsigned int j = 0; j < aBricks.size(); j++ ) {
      edm::Ptr< cmsUpgrades::L1TkStub< T > > stubToMatch = aBricks.at(j);
      unsigned int currentStack = stubToMatch->getStack();
      /// Speed-up
      /// Skip same layer Bricks
      if ( currentStack == innerStack || currentStack == outerStack ) continue;
      /// Get success of propagation
      bool isAccepted = false;
      int propSuccess = trkProp[ currentStack ]->getSuccess();
      /// Veto for gap between short barrels
      if ( currentStack >= 4 && currentStack <=7 ) {
        int propSuccessCompl = trkProp[ currentStack+6 ]->getSuccess();
        if ( propSuccess == 1 && propSuccessCompl != 1 ) isAccepted = true;
        /// Propagation must be successful for the long barrel
        /// but must fail for the gap between short barrels
      } /// End of Veto
      else { /// Full-length Long Barrel
        if ( propSuccess == 1 ) isAccepted = true;
      }

      /// Skip if bad propagation
      if ( isAccepted == false ) continue;

      /// Step 2.5: Prepare for coordinate comparison
      /// Select Detector Units
      GeomDetUnit* hitDetUnit;
      if ( currentStack > outerStack ) /// Forward
        hitDetUnit = (GeomDetUnit*)TrackingAlgorithm< T >::theStackedTracker->idToDetUnit( stubToMatch->getDetId(), 0 );
      else if ( currentStack < innerStack ) /// Backward
        hitDetUnit = (GeomDetUnit*)TrackingAlgorithm< T >::theStackedTracker->idToDetUnit( stubToMatch->getDetId(), 1 );
      else continue; /// Redundant cross-check

      bool isMatched = false;

      /// If we are here, it means the Seed propagation
      /// was successful, so we can mimic the PSimHit
      std::pair< double, PSimHit > hitProp = MakeHit( hitDetUnit, trkProp[ currentStack ], vertexCurvature );
      /// MakeHit acceptance
      if ( hitProp.first > 0 ) { /// Go on only if the first member of the pair
                                 /// is strictly positive
        /// Get the BRICK corresponding stub position
        GlobalPoint stubToMatchPosHits;
        if ( currentStack > outerStack ) /// Forward
          stubToMatchPosHits = stubToMatch->getCluster(0).getAveragePosition( TrackingAlgorithm< T >::theStackedTracker );
        else /// Back
          stubToMatchPosHits = stubToMatch->getCluster(1).getAveragePosition( TrackingAlgorithm< T >::theStackedTracker );
        /// Work for the match
        GlobalPoint curHitPosition = hitDetUnit->surface().toGlobal( hitProp.second.localPosition() );

        /// Get coordinates of corresponding Pixel, just as in
        /// SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h
        /// NOTE this is a very simplified version
        const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>( hitDetUnit );
        const PixelTopology* pixelTopol = dynamic_cast<const PixelTopology*>( &(pixelDet->specificTopology()));
        int numColumns = pixelTopol->ncolumns();  // det module number of cols&rows
        int numRows = pixelTopol->nrows();
        MeasurementPoint mpDigi = pixelTopol->measurementPosition( hitProp.second.localPosition() );
        int IPixRightUpX = int( floor( mpDigi.x()));
        int IPixRightUpY = int( floor( mpDigi.y()));
        int IPixLeftDownX = int( floor( mpDigi.x()));
        int IPixLeftDownY = int( floor( mpDigi.y()));
        IPixRightUpX = numRows>IPixRightUpX ? IPixRightUpX : numRows-1 ;
        IPixRightUpY = numColumns>IPixRightUpY ? IPixRightUpY : numColumns-1 ;
        IPixLeftDownX = 0<IPixLeftDownX ? IPixLeftDownX : 0 ;
        IPixLeftDownY = 0<IPixLeftDownY ? IPixLeftDownY : 0 ;
        float ix = 0.5*(IPixLeftDownX + IPixRightUpX);
        float iy = 0.5*(IPixLeftDownY + IPixRightUpY);
        mpDigi = MeasurementPoint (ix + 0.5, iy + 0.5);
        GlobalPoint curHitglobalPositionDigi = hitDetUnit->surface().toGlobal( pixelTopol->localPosition(mpDigi) );
        /// Correct for beamspot position
        GlobalPoint vertexBeamSpot = GlobalPoint( aSeed->getVertex().x(), aSeed->getVertex().y(), (double)0.0 );          
        //curHitglobalPosition = GlobalPoint( curHitglobalPosition.x() - vertexBeamSpot.x(),
        //                                    curHitglobalPosition.y() - vertexBeamSpot.y(),
        //                                    curHitglobalPosition.z() - vertexBeamSpot.z() );
        stubToMatchPosHits = GlobalPoint( stubToMatchPosHits.x() - vertexBeamSpot.x(),
                                          stubToMatchPosHits.y() - vertexBeamSpot.y(),
                                          stubToMatchPosHits.z() - vertexBeamSpot.z());
        curHitglobalPositionDigi = GlobalPoint( curHitglobalPositionDigi.x() - vertexBeamSpot.x(),
                                                curHitglobalPositionDigi.y() - vertexBeamSpot.y(),
                                                curHitglobalPositionDigi.z() - vertexBeamSpot.z());   

        /// Calculate displacement and load window size from the table
        double dZ = fabs( curHitglobalPositionDigi.z() - stubToMatchPosHits.z() );
        double dPhi = stubToMatchPosHits.phi() - curHitglobalPositionDigi.phi();
        if ( fabs(dPhi) >= cmsUpgrades::KGMS_PI) {
          if ( dPhi > 0 ) dPhi = dPhi - 2*cmsUpgrades::KGMS_PI;
          else dPhi = 2*cmsUpgrades::KGMS_PI - fabs(dPhi);
        }

        /// NOTE: DO NOT CHECK Z consistency of seed and brick tracklets
        /// information not available for Stubs
        /// CHECK MATCH
        if ( dZ <= windowsZ[ doubleStack ][ currentStack ] &&
             fabs(dPhi - biasPhi[ doubleStack ][ currentStack ]) <= windowsPhi[ doubleStack ][ currentStack ] ) isMatched = true;

        /// SAVE THE PAIR LATER ON, IF BOTH STUBS ARE WITHIN LIMITS
      } /// End of makeHit acceptance
      /// 
      if ( isMatched ) tempStubs[ currentStack ].push_back( stubToMatch );

    } /// End of loop over Bricks

    /// Define NULL pointer
    edm::Ptr< cmsUpgrades::L1TkStub< T > > nullPtr = edm::Ptr< cmsUpgrades::L1TkStub< T > >();

    /// Fill with NULL those empty vectors
    for ( unsigned int y = 0; y < 10; y++ )
      if ( tempStubs[y].size() == 0 ) tempStubs[y].push_back( nullPtr );

    /// Create all possible multiplets
    for ( unsigned int r0 = 0; r0 < tempStubs[0].size(); r0++ ) {
      for ( unsigned int r1 = 0; r1 < tempStubs[1].size(); r1++ ) {
        for ( unsigned int r2 = 0; r2 < tempStubs[2].size(); r2++ ) {
          for ( unsigned int r3 = 0; r3 < tempStubs[3].size(); r3++ ) {
            for ( unsigned int r4 = 0; r4 < tempStubs[4].size(); r4++ ) {
              for ( unsigned int r5 = 0; r5 < tempStubs[5].size(); r5++ ) {
                for ( unsigned int r6 = 0; r6 < tempStubs[6].size(); r6++ ) {
                  for ( unsigned int r7 = 0; r7 < tempStubs[7].size(); r7++ ) {
                    for ( unsigned int r8 = 0; r8 < tempStubs[8].size(); r8++ ) {
                      for ( unsigned int r9 = 0; r9 < tempStubs[9].size(); r9++ ) {
                        /// Fill the multiplet only with real Stubs
                        /// Fake Stubs with nullDet correspond only to
                        /// Stacks without any matched Stub in them
                        std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > tempChain;
                        tempChain.clear();
                        if ( tempStubs[0].at(r0) != nullPtr ) tempChain.push_back( tempStubs[0].at(r0) );
                        if ( tempStubs[1].at(r1) != nullPtr ) tempChain.push_back( tempStubs[1].at(r1) );
                        if ( tempStubs[2].at(r2) != nullPtr ) tempChain.push_back( tempStubs[2].at(r2) );
                        if ( tempStubs[3].at(r3) != nullPtr ) tempChain.push_back( tempStubs[3].at(r3) );
                        if ( tempStubs[4].at(r4) != nullPtr ) tempChain.push_back( tempStubs[4].at(r4) );
                        if ( tempStubs[5].at(r5) != nullPtr ) tempChain.push_back( tempStubs[5].at(r5) );
                        if ( tempStubs[6].at(r6) != nullPtr ) tempChain.push_back( tempStubs[6].at(r6) );
                        if ( tempStubs[7].at(r7) != nullPtr ) tempChain.push_back( tempStubs[7].at(r7) );
                        if ( tempStubs[8].at(r8) != nullPtr ) tempChain.push_back( tempStubs[8].at(r8) );
                        if ( tempStubs[9].at(r9) != nullPtr ) tempChain.push_back( tempStubs[9].at(r9) );
                        /// Here the Track constructor
                        cmsUpgrades::L1TkTrack< T > tempTrack( tempChain, aSeed );
                        tempTrackColl.push_back( tempTrack );
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } /// End of creation of all possible multiplets

    /// Return the collection of Tracks
    return tempTrackColl;

  } /// End of seed propagation method

  /// Make Hit from Propagated Seed in Detector Element
  template<typename T>
  std::pair< double, PSimHit > TrackingAlgorithm_bpphel< T >::MakeHit( const GeomDetUnit* dU, BaseParticlePropagator* tP, double curv ) const {

    /// Get position and momentum of propagated Seed
    math::XYZTLorentzVector globVtx = tP->vertex();
    math::XYZTLorentzVector globMom = tP->momentum();
    GlobalPoint  gpos = GlobalPoint(  globVtx.x(), globVtx.y(), globVtx.z() );
    GlobalVector gmom = GlobalVector(  globMom.x(), globMom.y(), globMom.z() );
    LocalPoint  lpos;
    LocalVector lmom;

    /// Something about detector element
    const float onSurfaceTolarance = 0.01; /// 10 microns
    GlobalPoint pCentDet = dU->toGlobal( dU->topology().localPosition( MeasurementPoint( 0.5*dU->surface().bounds().width(), 0.5*dU->surface().bounds().length() ) ) );

    /// Constraint in azimuthal wedge
    double deltaPhi = gpos.phi() - pCentDet.phi();
    if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI) {
      if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
      else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
    }
    deltaPhi = fabs(deltaPhi);
    if ( deltaPhi > 4.0*atan(1.0)/12.0 ) return std::pair< double, PSimHit >( -999.9, PSimHit() );

    PropagationDirection dirP = anyDirection;

    /// If the propagation position is close enough to the
    /// corresponding one on the detector in terms of z displacement,
    /// then use it to calculate the local position
    if ( fabs( dU->toLocal( gpos ).z() ) < onSurfaceTolarance ) {
      lpos = dU->toLocal( gpos );
      lmom = dU->toLocal( gmom );
    }
    else {
      HelixArbitraryPlaneCrossing crossing( gpos.basicVector(),
                                            gmom.basicVector(),
                                            curv,
                                            dirP );
      /// Check the impact is on the detector element
      std::pair< bool, double > path = crossing.pathLength( dU->surface() );
      if (!path.first)
        return std::pair< double, PSimHit >( -999.9, PSimHit() );
 
      /// Ok, here it is fine
      GlobalPoint gcrosspos = GlobalPoint( crossing.position(path.second) );
      lpos = dU->toLocal( gcrosspos );
      lmom = dU->toLocal( GlobalVector( crossing.direction(path.second) ) );
      lmom = lmom.unit() * (dU->toLocal( gmom )).mag();
    }

    /// Put together into a PSimHit
    /// The module (half) thickness 
    const BoundPlane& theDetPlane = dU->surface();
    float halfThick = 0.5*theDetPlane.bounds().thickness();
    /// The entry and exit points, and the time of flight
    float pZ = lmom.z();
    LocalPoint entry = lpos + (-halfThick/pZ) * lmom;
    LocalPoint exit = lpos + halfThick/pZ * lmom;
    float tof = gpos.mag() / 30. ; /// in nanoseconds, FIXME: very approximate
    PSimHit hit( entry, exit, lmom.mag(), tof, 0, -99999,
                 dU->geographicalId().rawId(), -99999,
                 lmom.theta(), lmom.phi() );

    /// Check that the PSimHit is physically on the module!
    double boundX = theDetPlane.bounds().width()/2.;  
    double boundY = theDetPlane.bounds().length()/2.;
    /// Check if the hit is on the physical volume of the module
    /// (It happens that it is not, in the case of double sided modules,
    ///  because the envelope of the gluedDet is larger than each of 
    ///  the mono and the stereo modules)
    double dist = 0.;
    GlobalPoint IP (0,0,0);
    dist = ( fabs(hit.localPosition().x()) > boundX  || 
             fabs(hit.localPosition().y()) > boundY ) ?  
    // Will be used later as a flag to reject the PSimHit!
    -( dU->surface().toGlobal(hit.localPosition()) - IP ).mag2() 
    : 
    // These hits are kept!
    ( dU->surface().toGlobal(hit.localPosition()) - IP ).mag2();
    /// NOTE THIS IS NOT IMPORTANT AS ONLY THE SIGN MATTERS
    return std::pair< double, PSimHit >( dist, hit );

  } /// End of Making the Hit


} /// Close namespace



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template<  typename T  >
class  ES_TrackingAlgorithm_bpphel: public edm::ESProducer{

  private:
    /// Data members
    boost::shared_ptr< cmsUpgrades::TrackingAlgorithm<T> > _theAlgo;
    double mWindowSize;
    edm::ParameterSet mParSet;

  public:
    /// Constructor
    ES_TrackingAlgorithm_bpphel( const edm::ParameterSet & p ) :
                                 mWindowSize( p.getParameter<double>("windowSize") ),
                                 mParSet( p ) /// This is to handle the table within the algorithm
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TrackingAlgorithm_bpphel() {}

    /// ///////////////// ///
    /// MANDATORY METHODS ///
    /// Implement the producer
    boost::shared_ptr< cmsUpgrades::TrackingAlgorithm<T> > produce( const cmsUpgrades::TrackingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle<MagneticField> magnet;
      record.getRecord<IdealMagneticFieldRecord>().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate B rounded to 4.0 or 3.8
      mMagneticFieldStrength = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
      cmsUpgrades::TrackingAlgorithm<T>* TrackingAlgo = new cmsUpgrades::TrackingAlgorithm_bpphel< T >( &(*StackedTrackerGeomHandle),
                                                                                                        mMagneticFieldStrength,
                                                                                                        mWindowSize,
                                                                                                        mParSet );

      _theAlgo  = boost::shared_ptr< cmsUpgrades::TrackingAlgorithm< T > >( TrackingAlgo );
      return _theAlgo;
    } 

};

#endif

