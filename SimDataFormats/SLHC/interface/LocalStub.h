/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June                           ///
///                                      ///
/// Added features:                      ///
/// Possibility to have a flag telling   ///
/// if the Stub is compatible with a     ///
/// higher Pt threshold than the default ///
/// one 'isTight()'. This is set in the  ///
/// StubBuilder. More details there.     ///
/// Possibility to have a flag telling   ///
/// if the Hits used to make the Stub    ///
/// are far away from sensor edges at    ///
/// least by 5 mm 'isInsideBorderRPhi()' ///
/// and 'isInsideBorderZ()'. This is set ///
/// in the StubBuilder. More details     ///
/// there.                               ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_LOCAL_STUB_FORMAT_H
#define STACKED_TRACKER_LOCAL_STUB_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"

//for the helper methods
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"

#include "DataFormats/Common/interface/Ref.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 


namespace cmsUpgrades{

template< typename T >
class LocalStub {

  typedef std::vector< T > HitCollection;

  public:

    typedef  std::set< std::pair<unsigned int, HitCollection> >  HitMap;
    typedef typename HitMap::const_iterator               HitMapIterator;

    LocalStub()
    {
      tightness = false;
      borderRPHI = false;
      borderZ = false;
      nullVector.clear();
      theHits.clear();
      id=0;
    }

    LocalStub(StackedTrackerDetId anId)
    {
      tightness = false;
      borderRPHI = false;
      borderZ = false;
      nullVector.clear();
      theHits.clear();
      id=anId;
    }


    ~LocalStub(){}

    /// Does the Stub pass a higher Pt threshold
    /// than the default one?
    bool isTight() const {
      return tightness;
    }
    void setTightness ( bool a ) {
      tightness = a;
    }

    /// Are hits far away from sensor edges?
    bool isInsideBorderRPhi() const {
      return borderRPHI;
    }
    void setInsideBorderRPhi ( bool a ) {
      borderRPHI = a;
    }
    bool isInsideBorderZ() const {
      return borderZ;
    }
    void setInsideBorderZ ( bool a ) {
      borderZ = a;
    }

    /// Already present in 2_2_X
    void addHit(   unsigned int point_within_hit ,
              const T &hitRef )
    {
      HitCollection temp;
      temp.push_back(hitRef);
      theHits.insert( std::make_pair( point_within_hit , temp ) );
    }

    void addCluster(  unsigned int point_within_hit , 
                const HitCollection &hitRefVector )
    {
      theHits.insert( std::make_pair( point_within_hit , hitRefVector ) ); 
    }


    const HitCollection &hit( unsigned int hitIdentifier ) const
    {
      for (HitMapIterator i = theHits.begin(); i != theHits.end(); ++i){
        if ( i->first == hitIdentifier ) return i->second;
      }
      return nullVector;

    }


    const HitMap& hits()
    {
      return theHits;
    }

    StackedTrackerDetId Id() const
    {
      return id;
    }

    //Helper Methods
    //GlobalPoint hitPosition(const GeomDetUnit* geom, const edm::Ref<edm::PSimHitContainer> &hit) const;
    GlobalPoint hitPosition(const GeomDetUnit* geom, const T &hit) const;

    GlobalPoint averagePosition( const cmsUpgrades::StackedTrackerGeometry *theStackedTracker , unsigned int hitIdentifier) const
    {
      double averageX = 0.0;
      double averageY = 0.0;
      double averageZ = 0.0;

      const  HitCollection &lhits = this->hit( hitIdentifier );
      typedef typename HitCollection::const_iterator IT;

      if(lhits.size()!=0){
        for (  IT hits_itr = lhits.begin();
            hits_itr != lhits.end();
            hits_itr++ )
        {
          const GeomDetUnit* det  = theStackedTracker->idToDetUnit( this->Id() , hitIdentifier );
  
          GlobalPoint thisHitPosition = hitPosition( det, *hits_itr );
          averageX += thisHitPosition.x();
          averageY += thisHitPosition.y();
          averageZ += thisHitPosition.z();
        }
        averageX /= lhits.size();
        averageY /= lhits.size();
        averageZ /= lhits.size();
      }
      return GlobalPoint(averageX, averageY, averageZ);
    }


    std::string print(unsigned int i=0) const {

      std::string padding("");
      for(unsigned int j=0;j!=i;++j)padding+="\t";

      std::stringstream output;
      output<<padding<<"LocalStub:\n";
      padding+='\t';
      output << padding << "StackedTrackerDetId: " << id << '\n';

      for( HitMapIterator it=theHits.begin() ; it!=theHits.end() ; ++it )
        output << padding << "member " << it->first << ", cluster size = " << it->second.size() << '\n';
      return output.str();
    }


  private:
    bool tightness;
    bool borderRPHI;
    bool borderZ;
    HitMap theHits;
    StackedTrackerDetId id;
    HitCollection nullVector;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Now for the implimentation of the helper methods
//Specialize the template for PSimHits
  template<> GlobalPoint cmsUpgrades::LocalStub< edm::Ref<edm::PSimHitContainer> >::hitPosition(const GeomDetUnit* geom, const edm::Ref<edm::PSimHitContainer> &hit) const;

//Default assumes pixelization
  template<  typename T  >
  GlobalPoint cmsUpgrades::LocalStub<T>::hitPosition(const GeomDetUnit* geom, const T &hit) const
  {
    MeasurementPoint mp( hit->row() + 0.5, hit->column() + 0.5 ); // Add 0.5 to get the center of the pixel.
    return geom->surface().toGlobal( geom->topology().localPosition( mp ) ) ;
  }
//////////////////////////////////////////////////////////////////////////////////////////////////////////
}

template<  typename T  >
std::ostream& operator << (std::ostream& os, const cmsUpgrades::LocalStub< T >& aLocalStub) {
  return (os<<aLocalStub.print() );
//  return os;
}

#endif






