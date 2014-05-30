/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2010, June                           ///
/// 2011, June                           ///
/// 2013, January                        ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_TRACK_FORMAT_H
#define STACKED_TRACKER_L1TK_TRACK_FORMAT_H

#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/SLHC/interface/L1TkStub.h"

/** ************************ **/
/**                          **/
/**   DECLARATION OF CLASS   **/
/**                          **/
/** ************************ **/

template< typename T >
class L1TkTrack
{
  private:
    /// Data members
    std::vector< edm::Ptr< L1TkStub< T > > >  theStubPtrs;
    GlobalVector                              theMomentum;
    GlobalPoint                               theVertex;
    double                                    theRInv;
    unsigned int                              theSector;
    unsigned int                              theWedge;
    double                                    theChi2;
    edm::Ptr< SimTrack >                      theSimTrack;
    uint32_t                                  theEventId;

  public:
    /// Constructors
    L1TkTrack();
    L1TkTrack( std::vector< edm::Ptr< L1TkStub< T > > > aStubs );

    /// Destructor
    ~L1TkTrack();

    /// Track components
    std::vector< edm::Ptr< L1TkStub< T > > > getStubPtrs() const { return theStubPtrs; }
    void addStubPtr( edm::Ptr< L1TkStub< T > > aStub );

    /// Track momentum
    GlobalVector getMomentum() const { return theMomentum; }
    void         setMomentum( GlobalVector aMomentum );

    /// Track parameters
    double getRInv() const { return theRInv; }
    void   setRInv( double aRInv );

    /// Vertex
    GlobalPoint getVertex() const { return theVertex; }
    void        setVertex( GlobalPoint aVertex );

    /// Sector
    unsigned int getSector() const { return theSector; }
    void         setSector( unsigned int aSector );
    unsigned int getWedge() const { return theWedge; }
    void         setWedge( unsigned int aWedge );

    /// Chi2
    double getChi2() const { return theChi2; }
    double getChi2Red() const;
    void   setChi2( double aChi2 );

    /// MC Truth
    edm::Ptr< SimTrack > getSimTrackPtr() const { return theSimTrack; }
    uint32_t             getEventId() const { return theEventId; }
    bool                 isGenuine() const;
    bool                 isCombinatoric() const;
    bool                 isUnknown() const;
    int                  findType() const;
    unsigned int         findSimTrackId() const;

    /// Superstrip
    /// Here to prepare inclusion of AM L1 Track finding
    uint32_t getSuperStrip() const { return 0; }

    /// ////////////// ///
    /// HELPER METHODS ///
    bool isTheSameAs( L1TkTrack< T > aTrack ) const;

    /// Fake or not
    void checkSimTrack();

      /// Tricky Fit as suggested by Pierluigi, employing
      /// a tracklet-style approach for triplets within
      /// the chain of stubs composing the track
//      void fitTrack( double aMagneticFieldStrength, bool useAlsoVtx, bool aDoHelixFit );

      /// /////////////////// ///
      /// INFORMATIVE METHODS ///
//      std::string print( unsigned int i=0 ) const;

}; /// Close class

/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/
 
/// Default Constructor
template< typename T >
L1TkTrack< T >::L1TkTrack()
{
  theStubPtrs.clear();
  theMomentum = GlobalVector(0.0,0.0,0.0);
  theRInv     = 0;
  theVertex   = GlobalPoint(0.0,0.0,0.0);
  theSector   = 0;
  theWedge    = 0;
  theChi2     = 0;
  /// theSimTrack is NULL by default
  theEventId = 0xFFFF;
}

/// Another Constructor
template< typename T >
L1TkTrack< T >::L1TkTrack( std::vector< edm::Ptr< L1TkStub< T > > > aStubs )
{
  theStubPtrs = aStubs;
  theMomentum = GlobalVector(0.0,0.0,0.0);
  theVertex   = GlobalPoint(0.0,0.0,0.0);
  theRInv     = 0;
  theSector   = 0;
  theWedge    = 0;
  theChi2     = 0;
  /// theSimTrack is NULL by default
  theEventId = 0xFFFF;
}

/// Destructor
template< typename T >
L1TkTrack< T >::~L1TkTrack(){}

/// Track components
template< typename T >
void L1TkTrack< T >::addStubPtr( edm::Ptr< L1TkStub< T > > aStub )
{
  theStubPtrs.push_back( aStub );
}

/// Track momentum
template< typename T >
void L1TkTrack< T >::setMomentum( GlobalVector aMomentum )
{
  theMomentum = aMomentum;
}

/// Vertex
template< typename T >
void L1TkTrack< T >::setVertex( GlobalPoint aVertex )
{
  theVertex = aVertex;
}

/// Track parameters
template< typename T >
void L1TkTrack< T >::setRInv( double aRInv )
{
  theRInv = aRInv;
}

/// Sector
template< typename T >
void L1TkTrack< T >::setSector( unsigned int aSector )
{
  theSector = aSector;
}

template< typename T >
void L1TkTrack< T >::setWedge( unsigned int aWedge )
{
  theWedge = aWedge;
}

/// Chi2
template< typename T >
double L1TkTrack< T >::getChi2Red() const { return theChi2/( 2*theStubPtrs.size() - 4 ); }

template< typename T >
void L1TkTrack< T >::setChi2( double aChi2 )
{
  theChi2 = aChi2;
}

// MC truth
template< typename T >
int L1TkTrack< T >::findType() const
{
  if ( theSimTrack.isNull() )
    return 999999999;
  return theSimTrack->type();
}

template< typename T >
unsigned int L1TkTrack< T >::findSimTrackId() const
{
  if ( theSimTrack.isNull() )
    return 0;
  return theSimTrack->trackId();
}

/// ////////////// ///
/// HELPER METHODS ///
/// ////////////// ///

/// Check if two tracks are the same
template< typename T>
bool L1TkTrack< T >::isTheSameAs( L1TkTrack< T > aTrack ) const
{
  /// Take the other stubs
  std::vector< edm::Ptr< L1TkStub< T > > > otherStubPtrs = aTrack.getStubPtrs();

  /// Count shared stubs
  unsigned int nShared = 0;
  for ( unsigned int i = 0; i < theStubPtrs.size() && nShared < 2; i++)
  {
    for ( unsigned int j = 0; j < otherStubPtrs.size() && nShared < 2; j++)
    {
      if ( theStubPtrs.at(i) == otherStubPtrs.at(j) )
      {
        nShared++;
      }
    }
  }

  /// Same track if 2 shared stubs
  return ( nShared > 1 );
}

/// Check SimTracks
template< typename T >
void L1TkTrack< T >::checkSimTrack()
{
  /// Vector to store SimTracks
  std::vector< edm::Ptr< SimTrack > > tempVecGen;
  std::vector< std::vector< edm::Ptr< SimTrack > > > tempVecComb;
  std::vector< uint32_t >                evIdGen;
  std::vector< std::vector< uint32_t > > evIdComb;

  /// Loop over the Stubs
  /// Put the SimTracks together
  /// SimTracks from genuine stubs in one container
  /// SimTracks from combinatoric stubs in another container
  for ( unsigned int js = 0; js < theStubPtrs.size(); js++ )
  {
    /// If one Stub is unknown, also the Track is unknown
    if ( theStubPtrs.at(js)->isUnknown() )
      return;

    /// Store SimTracks for non-unknown Stubs
    if ( theStubPtrs.at(js)->isGenuine() )
    {
      tempVecGen.push_back( theStubPtrs.at(js)->getSimTrackPtr() );
      evIdGen.push_back( theStubPtrs.at(js)->getEventId() );
    }
    else if ( theStubPtrs.at(js)->isCombinatoric() )
    {
      for ( unsigned int ic = 0; ic < 2; ic++ )
      {
        std::vector< edm::Ptr< SimTrack > > cluVec = theStubPtrs.at(js)->getClusterPtrs().at(ic)->getSimTrackPtrs();
        tempVecComb.push_back( cluVec );
        std::vector< uint32_t > evVec = theStubPtrs.at(js)->getClusterPtrs().at(ic)->getEventIds();
        evIdComb.push_back( evVec );
      }
    }
  }

  if ( 2*tempVecGen.size() + tempVecComb.size() != 2*theStubPtrs.size() )
  {
    std::cerr << "ERROR!!! HERE WE ARE SUPPOSED TO HAVE ONLY GENUINE AND COMBINATORIAL STUBS" << std::endl;
    std::cerr << theStubPtrs.size() << " = " << tempVecGen.size() << " + " << tempVecComb.size()/2 << std::endl;
    return;
  }

  if ( tempVecComb.size() % 2 != 0 )
  {
    std::cerr << "ERROR!!! NO ODD NUMBER OF CLUSTERS FROM COMBINATORIAL STUBS IS SUPPOSED TO BE" << std::endl;
    return;
  }

  /// If we got here, it means that all the Stubs are genuine/combinatoric
  /// COMBINATORIC means that no common SimTrack can be found
  /// GENUINE means otherwise
  int idSimTrackG = -99999;
  uint32_t evIdG = 0xFFFF;

  if ( tempVecGen.size() > 0 )
  {
    /// Case of >=1 genuine Stubs
    idSimTrackG = tempVecGen.at(0)->trackId();
    evIdG = evIdGen.at(0);

    for ( unsigned int jg = 1; jg < tempVecGen.size(); jg++ )
    {
      /// Two genuine Stubs with different EventId mean COMBINATORIC
      if ( evIdGen.at(0) != evIdG )
        return;

      /// Two genuine Stubs with different SimTrack mean COMBINATORIC
      if ( (int)(tempVecGen.at(0)->trackId()) != idSimTrackG )
        return;
    }

    /// If we got here, it means that all the genuine Stubs have the same SimTrack
    /// Time to check the combinatoric ones

    /// Case of no combinatoric Stubs
    if ( tempVecComb.size() == 0 )
    {
      /// No combinatoric stubs found
      /// All genuine, all the same SimTrack
      theSimTrack = tempVecGen.at(0);
      theEventId = evIdGen.at(0);
      return;
    }

    /// Case of at least 1 Stub is combinatoric
    /// If we are here, we must have EVEN tempVecComb.size()
    for ( unsigned int jc1 = 0; jc1 < tempVecComb.size(); jc1+=2 )
    {
      bool foundSimTrack = false;
      /// Check first cluster (see how they are pushed_back)
      for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(jc1).size() && !foundSimTrack; jc0++ )
      {
        if ( tempVecComb.at(jc1).at(jc0).isNull() )
          continue;

        if ( (int)(tempVecComb.at(jc1).at(jc0)->trackId()) == idSimTrackG &&
             evIdComb.at(jc1).at(jc0) == evIdG )
        {
          foundSimTrack = true;
        }
      }
      /// Check second cluster
      for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(jc1+1).size() && !foundSimTrack; jc0++ )
      {
        if ( tempVecComb.at(jc1+1).at(jc0).isNull() )
          continue;

        if ( (int)(tempVecComb.at(jc1+1).at(jc0)->trackId()) == idSimTrackG &&
             evIdComb.at(jc1+1).at(jc0) == evIdG )
        {
          foundSimTrack = true;
        }
      }

      if ( !foundSimTrack )
        return;

      /// If we got here, we have >= 1 genuine Stub whose SimTrack
      /// is found in at least 1 Cluster of all other Stubs
      theSimTrack = tempVecGen.at(0);
      theEventId = evIdGen.at(0);
      return;
    }
  }
  else
  {
    /// No genuine Stubs are found
    if ( tempVecComb.size() == 0 )
    {
      std::cerr << "YOU SHOULD NEVER GET HERE (0 Genuine and 0 Combinatoric Stubs in a Track)" << std::endl;
      return;
    }
    else if ( tempVecComb.size() == 1 )
    {
      std::cerr << "YOU SHOULD NEVER GET HERE (0 Genuine and 1 Combinatoric Stubs in a Track)" << std::endl;
      return;
    }

    /// We have only combinatoric Stubs
    /// We need to have the same SimTrack in all Stubs
    /// If we are here, we must have EVEN tempVecComb.size()
    /// Map by SimTrackId all the SimTracks and count in how many Stubs they are
    std::map< std::pair< unsigned int, uint32_t >, std::vector< unsigned int > > mapSimTrack;

    for ( unsigned int jc1 = 0; jc1 < tempVecComb.size(); jc1+=2 )
    {
      /// Check first cluster (see how they are pushed_back)
      for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(jc1).size(); jc0++ )
      {
        if ( tempVecComb.at(jc1).at(jc0).isNull() )
          continue;

        std::pair< unsigned int, uint32_t > thisId = std::make_pair( tempVecComb.at(jc1).at(jc0)->trackId(), evIdComb.at(jc1).at(jc0) );

        if ( mapSimTrack.find( thisId ) == mapSimTrack.end() )
        {
          /// New SimTrack
          /// Push back which Stub idx it is within the Combinatorial ones
          std::vector< unsigned int > tempVecStubIdx;
          tempVecStubIdx.push_back( jc1/2 );
          mapSimTrack.insert( std::make_pair( thisId, tempVecStubIdx ) );
        }
        else
        {
          /// Existing SimTrack
          /// Push back which Stub idx it is within the Combinatorial ones
          mapSimTrack.find( thisId )->second.push_back( jc1/2 );
        }
      }
      /// Check second cluster
      for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(jc1+1).size(); jc0++ )
      {
        if ( tempVecComb.at(jc1+1).at(jc0).isNull() )
          continue;

        std::pair< unsigned int, uint32_t > thisId = std::make_pair( tempVecComb.at(jc1+1).at(jc0)->trackId(), evIdComb.at(jc1+1).at(jc0) );
        
        if ( mapSimTrack.find( thisId ) == mapSimTrack.end() )
        {
          /// New SimTrack
          /// Push back which Stub idx it is within the Combinatorial ones
          std::vector< unsigned int > tempVecStubIdx;
          tempVecStubIdx.push_back( jc1/2 );
          mapSimTrack.insert( std::make_pair( thisId, tempVecStubIdx ) );
        }
        else
        {
          /// Existing SimTrack
          /// Push back which Stub idx it is within the Combinatorial ones
          mapSimTrack.find( thisId )->second.push_back( jc1/2 );
        }
      }
    }

    /// Check the SimTrack Map
    unsigned int countSimTracks = 0;
    unsigned int theSimTrackId = 0;
    uint32_t     theStoredEventId = 0xFFFF;
    std::map< std::pair< unsigned int, uint32_t >, std::vector< unsigned int > >::iterator mapIt;
    for ( mapIt = mapSimTrack.begin();
          mapIt != mapSimTrack.end();
          ++mapIt )
    {
      /// SimTracks found in 1 Cluster of ALL stubs
      /// This means that counting the number of different occurencies in the
      /// vector one should get the size of the vector of Stubs in the track
      /// So, sort and remove duplicates
      std::vector< unsigned int > tempVector = mapIt->second;
      std::sort( tempVector.begin(), tempVector.end() );
      tempVector.erase( std::unique( tempVector.begin(), tempVector.end() ), tempVector.end() );

      if ( tempVector.size() == theStubPtrs.size() )
      {
        countSimTracks++;
        theSimTrackId = mapIt->first.first;
        theStoredEventId = mapIt->first.second;
      }
    }

    /// We want only 1 SimTrack!
    if ( countSimTracks != 1 )
      return;

    /// We can look for the SimTrack now ...
    /// By construction, it must be in the first Stub ...
    /// Check first cluster (see how they are pushed_back)
    for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(0).size(); jc0++ )
    {
      if ( tempVecComb.at(0).at(jc0).isNull() )
        continue;

      if ( theSimTrackId == tempVecComb.at(0).at(jc0)->trackId() &&
           theStoredEventId == evIdComb.at(0).at(jc0) )
      {
        theSimTrack = tempVecComb.at(0).at(jc0);
        theEventId = evIdComb.at(0).at(jc0);
        return;
      }
    }

    /// Check second cluster
    for ( unsigned int jc0 = 0; jc0 < tempVecComb.at(1).size(); jc0++ )
    {
      if ( tempVecComb.at(1).at(jc0).isNull() )
        continue;

      if ( theSimTrackId == tempVecComb.at(1).at(jc0)->trackId() &&
           theStoredEventId == evIdComb.at(1).at(jc0) )
      {
        theSimTrack = tempVecComb.at(1).at(jc0);
        theEventId = evIdComb.at(1).at(jc0);
        return;
      }
    }
  } /// End of no genuine stubs are found

  std::cerr << "YOU SHOULD NEVER GET HERE (all cases MUST have been processed earlier)" << std::endl;

}

template< typename T >
bool L1TkTrack< T >::isUnknown() const
{
  /// UNKNOWN means that at least 1 Stub is UNKNOWN

  /// Loop over the stubs
  for ( unsigned int js = 0; js < theStubPtrs.size(); js++ )
  {
    /// If one Stub is unknown, also the Track is unknown
    if ( theStubPtrs.at(js)->isUnknown() )
      return true;
  }
  return false;
}


template< typename T >
bool L1TkTrack< T >::isGenuine() const
{
  /// GENUINE means that we could set a SimTrack
  if ( theSimTrack.isNull() == false )
    return true;

  return false;
}


template< typename T >
bool L1TkTrack< T >::isCombinatoric() const
{
  if ( this->isGenuine() )
    return false;

  if ( this->isUnknown() )
    return false;

  return true;
}



/*
  /// Fit
  template< typename T >
  void L1TkTrack< T >::fitTrack( double aMagneticFieldStrength, bool useAlsoVtx, bool aDoHelixFit )
  {
    /// Step 00
    /// Get the magnetic field
    /// Usually it is done like the following three lines
    //iSetup.get<IdealMagneticFieldRecord>().get(magnet);
    //magnet_ = magnet.product();
    //mMagneticFieldStrength = magnet_->inTesla(GlobalPoint(0,0,0)).z();
    /// Calculate factor for rough Pt estimate
    /// B rounded to 4.0 or 3.8
    /// This is B * C / 2 * appropriate power of 10
    /// So it's B * 0.0015
    double mPtFactor = (floor(aMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015;

    /// Step 0
    /// Get Stubs chain and Vertex      
    std::vector< L1TkStub< T > > brickStubs = this->getStubs();
    L1TkTracklet< T >            seedTracklet = this->getSeedTracklet();
    /// This automatically sets 00 or beamspot according to L1TkTracklet type
    GlobalPoint seedVertexXY = GlobalPoint( seedTracklet.getVertex().x(), seedTracklet.getVertex().y(), 0.0 );

    /// If the seed vertex is requested for the fit, add it to stubs
    if ( useAlsoVtx ) {
      /// Prepare dummy stub with vertex position
      std::vector< L1TkStub< T > > auxStubs;
      auxStubs.clear();
      L1TkStub< T > dummyStub = L1TkStub< T >( 0 );

//      dummyStub.setPosition( seedTracklet.getVertex() );
//      dummyStub.setDirection( GlobalVector(0,0,0) );

      auxStubs.push_back( dummyStub );
      /// Put together also other stubs
      for ( unsigned int j = 0; j < brickStubs.size(); j++ ) auxStubs.push_back( brickStubs.at(j) );
      /// Overwrite
      brickStubs = auxStubs;
    }
*/



/*
    /// Step 1
    /// Find charge using only stubs, regardless of useAlsoVtx option!
    unsigned int iMin = 0;
    if ( useAlsoVtx ) iMin = 1;
    /// Check L1TkTracklet for further information
    double outerPointPhi = brickStubs.at( brickStubs.size()-1 ).getPosition().phi();
    double innerPointPhi = brickStubs.at( iMin ).getPosition().phi();
    double deltaPhi = outerPointPhi - innerPointPhi;
    if ( fabs(deltaPhi) >= KGMS_PI) {
      if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*KGMS_PI;
      else deltaPhi = 2*KGMS_PI - fabs(deltaPhi);
    }
    double deltaPhiC = deltaPhi; /// This is for charge
    deltaPhi = fabs(deltaPhi);
    double fCharge = -deltaPhiC / deltaPhi;
    this->setCharge( fCharge );

    /// Step 2
    /// Average for Momentum and Axis
    std::vector< double > outputFitPt; outputFitPt.clear();
    std::vector< double > outputFitPz; outputFitPz.clear();
    std::vector< double > outputFitX;  outputFitX.clear();
    std::vector< double > outputFitY;  outputFitY.clear();
    /// Now loop over Triplets
    unsigned int totalTriplets = 0;
    for ( unsigned int a1 = 0; a1 < brickStubs.size(); a1++ ) {
      for ( unsigned int a2 = a1+1; a2 < brickStubs.size(); a2++ ) {
        for ( unsigned int a3 = a2+1; a3 < brickStubs.size(); a3++ ) {
          totalTriplets++;
          /// Read Stubs in a "L1TkTracklet-wise" way
          GlobalPoint vtxPos = brickStubs.at(a1).getPosition();
          GlobalPoint innPos = brickStubs.at(a2).getPosition();
          GlobalPoint outPos = brickStubs.at(a3).getPosition();
          /// Correct for position of a1
          innPos = GlobalPoint( innPos.x()-vtxPos.x(), innPos.y()-vtxPos.y(), innPos.z() );
          outPos = GlobalPoint( outPos.x()-vtxPos.x(), outPos.y()-vtxPos.y(), outPos.z() );
          double outRad = outPos.perp();
          double innRad = innPos.perp();
          deltaPhi = outPos.phi() - innPos.phi(); /// NOTE overwrite already declared deltaPhi
          if ( fabs(deltaPhi) >= KGMS_PI ) {
            if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*KGMS_PI;
            else deltaPhi = 2*KGMS_PI - fabs(deltaPhi);
          }
          deltaPhi = fabs(deltaPhi);
          double x2 = outRad * outRad + innRad * innRad - 2 * innRad * outRad * cos(deltaPhi);
          double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));
          double roughPt = mPtFactor * twoRadius;
          double roughPz;
          /// Switch fit type
          if ( !aDoHelixFit ) roughPz = roughPt * (outPos.z()-innPos.z()) / (outRad-innRad);
          else {
            double phioi = acos(1 - 2*x2/(twoRadius*twoRadius));
            if ( fabs(phioi) >= KGMS_PI ) {
              if ( phioi>0 ) phioi = phioi - 2*KGMS_PI;
              else phioi = 2*KGMS_PI - fabs(phioi);
            }
            if ( phioi == 0 ) return;
            roughPz = 2 * mPtFactor * (outPos.z()-innPos.z()) / fabs(phioi);
          }
          /// Store Momenta for average
          outputFitPt.push_back( roughPt );
          outputFitPz.push_back( roughPz );
          /// Find angle from a1 pointing to Axis
          double vertexangle = acos( outRad/twoRadius );
          vertexangle = outPos.phi() - fCharge * vertexangle;
          /// Helix axis
          outputFitX.push_back( 0.5 * twoRadius * cos(vertexangle) + vtxPos.x() );
          outputFitY.push_back( 0.5 * twoRadius * sin(vertexangle) + vtxPos.y() );
        } /// End of loop over third element
      } /// End of loop over second element
    } /// End of loop over first element
    /// Compute averages and store them
    double tempOutputX = 0;
    double tempOutputY = 0;
    double tempOutputPt = 0;
    double tempOutputPz = 0;
    for ( unsigned int q = 0; q < totalTriplets; q++ ) {
      tempOutputX += outputFitX.at(q);
      tempOutputY += outputFitY.at(q);
      tempOutputPt += outputFitPt.at(q);
      tempOutputPz += outputFitPz.at(q);        
    }

    /// Step 3
    /// Get Helix Axis and correct wrt Seed VTX
    GlobalPoint fAxis = GlobalPoint( tempOutputX/totalTriplets, tempOutputY/totalTriplets, 0.0 );
    GlobalPoint fAxisCorr = GlobalPoint( fAxis.x() - seedVertexXY.x(), fAxis.y() - seedVertexXY.y(), 0.0 );
    this->setAxis( tempOutputX/totalTriplets, tempOutputY/totalTriplets );

    /// Step 4
    /// Momentum, starting from azimuth at vertex
    double fPhiV = atan2( fCharge*fAxisCorr.x(), -fCharge*fAxisCorr.y() );
    double fPt = tempOutputPt/totalTriplets;
    double fPz = tempOutputPz/totalTriplets;
    double fRadius  = 0.5*fPt/mPtFactor;
    GlobalVector fMomentum = GlobalVector( cos(fPhiV)*fPt, sin(fPhiV)*fPt, fPz );
    this->setMomentum( fMomentum );

    /// Step 5
    /// Average for Vertex (Closest Approach)
    double rMinAppr = fAxisCorr.perp() - fRadius;
    double xMinAppr = rMinAppr*cos( fAxisCorr.phi() ) + seedVertexXY.x();
    double yMinAppr = rMinAppr*sin( fAxisCorr.phi() ) + seedVertexXY.y();
    GlobalPoint tempVertex = GlobalPoint( xMinAppr, yMinAppr, 0.0 );
    double propFactorHel = 0;
    double offsetHel = 0;
    /// Average for Vtx z
    std::vector< double > outputFitZ; outputFitZ.clear();
    /// Now loop over Doublets
    /// Cannot put into the same loop as before because
    /// here we need radius and therefore Pt to have
    /// the radius, and the radius is needed to find the
    /// closest approach distance, is it clear?
    unsigned int totalDoublets = 0;
    for ( unsigned int a1 = iMin; a1 < brickStubs.size(); a1++) { /// iMin already set according to useAlsoVtx or not
      for ( unsigned int a2 = a1+1; a2 < brickStubs.size(); a2++) {
        totalDoublets++;
        /// Read Stubs in a "L1TkTracklet-wise" way
        GlobalPoint innPos = brickStubs.at(a1).getPosition();
        GlobalPoint outPos = brickStubs.at(a2).getPosition();
        /// Calculate z = z0 + c*phiStar
        GlobalPoint innPosStar = GlobalPoint( innPos.x() - fAxis.x(), innPos.y() - fAxis.y(), innPos.z() - fAxis.z() );
        GlobalPoint outPosStar = GlobalPoint( outPos.x() - fAxis.x(), outPos.y() - fAxis.y(), outPos.z() - fAxis.z() );
        double deltaPhiStar = outPosStar.phi() - innPosStar.phi();
        if ( fabs(deltaPhiStar) >= KGMS_PI ) {
          if ( outPosStar.phi() < 0 ) deltaPhiStar += KGMS_PI;
          else deltaPhiStar -= KGMS_PI;
          if ( innPosStar.phi() < 0 ) deltaPhiStar -= KGMS_PI;
          else deltaPhiStar += KGMS_PI;
        }
        if ( deltaPhiStar == 0 ) std::cerr<<"BIG PROBLEM IN DELTAPHI DENOMINATOR"<<std::endl;
        else {
          propFactorHel += ( outPosStar.z() - innPosStar.z() )/deltaPhiStar;
          offsetHel += innPosStar.z() - innPosStar.phi()*( outPosStar.z() - innPosStar.z() )/deltaPhiStar;
        } /// End of calculate z = z0 + c*phiStar
        innPos = GlobalPoint( innPos.x() - tempVertex.x(), innPos.y() - tempVertex.y(), innPos.z() );
        outPos = GlobalPoint( outPos.x() - tempVertex.x(), outPos.y() - tempVertex.y(), outPos.z() );
        double outRad = outPos.perp();
        double innRad = innPos.perp();
        deltaPhi = outPos.phi() - innPos.phi(); /// NOTE overwrite already declared deltaPhi
        if ( fabs(deltaPhi) >= KGMS_PI ) {
          if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*KGMS_PI;
          else deltaPhi = 2*KGMS_PI - fabs(deltaPhi);
        }
        deltaPhi = fabs(deltaPhi);
        double x2 = outRad * outRad + innRad * innRad - 2 * innRad * outRad * cos(deltaPhi);
        double twoRadius = sqrt(x2) / sin(fabs(deltaPhi));
        double zProj;
        /// Switch fit type
        if ( !aDoHelixFit ) zProj = outPos.z() - ( outRad * (outPos.z() - innPos.z()) / (outRad - innRad) );
        else {
          double phioi = acos(1 - 2*x2/(twoRadius*twoRadius));
          double phiiv = acos(1 - 2*innRad*innRad/(twoRadius*twoRadius));
          if ( fabs(phioi) >= KGMS_PI ) {
            if ( phioi>0 ) phioi = phioi - 2*KGMS_PI;
            else phioi = 2*KGMS_PI - fabs(phioi);
          }
          if ( fabs(phiiv) >= KGMS_PI ) {
            if ( phiiv>0 ) phiiv = phiiv - 2*KGMS_PI;
            else phiiv = 2*KGMS_PI - fabs(phiiv);
          }
          if ( phioi == 0 ) return;
          /// Vertex
          zProj = innPos.z() - (outPos.z()-innPos.z())*phiiv/phioi;
        }
        outputFitZ.push_back( zProj );
      } /// End of loop over second element
    } /// End of loop over first element
    /// Compute averages and store them
    double tempOutputZ = 0;
    for ( unsigned int q = 0; q < totalDoublets; q++ ) tempOutputZ += outputFitZ.at(q);
    double zMinAppr = tempOutputZ/totalDoublets;

    /// Step 6
    /// Vertex
    GlobalPoint fVertex = GlobalPoint( xMinAppr, yMinAppr, zMinAppr );
    this->setVertex( fVertex );


    /// Step 7
    /// Calculate Chi2
    propFactorHel = propFactorHel/totalDoublets;
    offsetHel = offsetHel/totalDoublets;
    double fChi2RPhi = 0;
    double fChi2ZPhi = 0;
    double tempStep;
    /// Calculate for the Seed VTX if needed
    if (useAlsoVtx) {
      GlobalPoint posPoint = seedTracklet.getVertex();
      GlobalPoint posPointCorr = GlobalPoint( posPoint.x() - fAxis.x(), posPoint.y() - fAxis.y(), posPoint.z() );
      /// Add X: x_meas - x_fit(phi*_meas)
      tempStep = posPoint.x() - fAxis.x() - fRadius * cos( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      /// Add Y: y_meas - y_fit(phi*_meas)
      tempStep = posPoint.y() - fAxis.y() - fRadius * sin( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      /// b = propFactorH
      /// a = offsetH
      /// z = b*phi - a
      /// Add Z: z_meas - z_fit(phi*_meas)
      tempStep = posPoint.z() - offsetHel - propFactorHel * posPointCorr.phi();
      fChi2ZPhi += tempStep*tempStep;
    }
    /// Calculate for all other Stubs
    for ( unsigned int a = iMin; a < brickStubs.size(); a++ ) {
      GlobalPoint posPoint = brickStubs.at(a).getPosition();
      GlobalPoint posPointCorr = GlobalPoint( posPoint.x() - fAxis.x(), posPoint.y() - fAxis.y(), posPoint.z() );

      tempStep = posPoint.x() - fAxis.x() - fRadius * cos( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;
      tempStep = posPoint.y() - fAxis.y() - fRadius * sin( posPointCorr.phi() );
      fChi2RPhi += tempStep*tempStep;

      tempStep = posPoint.z() - offsetHel - propFactorHel * posPointCorr.phi();
      fChi2ZPhi += tempStep*tempStep;
    }
    this->setChi2RPhi( fChi2RPhi );
    this->setChi2ZPhi( fChi2ZPhi );

*/



/*
  }



  /// /////////////////// ///
  /// INFORMATIVE METHODS ///
  /// /////////////////// ///

  template< typename T >
  std::string L1TkTrack< T >::print( unsigned int i ) const {
    std::string padding("");
    for ( unsigned int j=0; j!=i; ++j )padding+="\t";
    std::stringstream output;
    output<<padding<<"L1TkTrack:\n";
    padding+='\t';
    output << padding << "SeedDoubleStack: " << this->getSeedDoubleStack() << '\n';
    output << padding << "Length of Chain: " << theBrickStubs.size() << '\n';
    unsigned int iStub = 0;
    for ( L1TkStubPtrCollectionIterator i = theBrickStubs.begin(); i!= theBrickStubs.end(); ++i )
      output << padding << "stub: " << iStub++ << ", stack: \n";// << (*i)->getStack() << ", rough Pt: " << '\n';//(*i)->getRoughPt() << '\n';
    return output.str();
  }

  template< typename T >
  std::ostream& operator << (std::ostream& os, const L1TkTrack< T >& aL1TkTrack) {
    return (os<<aL1TkTrack.print() );
  }
*/













#endif


