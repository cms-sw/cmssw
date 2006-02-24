#ifdef PROJECT_NAME
#include "RecoVertex/VertexTools/interface/SMS.h"
#else
#include <vector>
#include <algo.h>
#include "RecoVertex/VertexTools/interface/SMS.h"
using namespace std;
#endif

namespace {


  typedef pair < float, const GlobalPoint * > MyPair;
  typedef pair < float, float > FloatPair;
  typedef pair < GlobalPoint, float > GlPtWt;
  typedef pair < float, const GlPtWt * > MyPairWt;

  struct Sorter
  {
    bool operator() ( const MyPair & pair1, const MyPair & pair2 )
    {
      return ( pair1.first < pair2.first );
    };

    bool operator() ( const FloatPair & pair1, const FloatPair & pair2 )
    {
      return ( pair1.first < pair2.first );
    };

    bool operator() ( const MyPairWt & pair1, const MyPairWt & pair2 )
    {
      return ( pair1.first < pair2.first );
    };
  };

  bool debug()
  {
    return false;
  };

  inline GlobalPoint & operator += ( GlobalPoint & a, const GlobalPoint & b )
  {
    a = GlobalPoint ( a.x() + b.x(), a.y() + b.y(), a.z() + b.z() );
    return a;
  };
  inline GlobalPoint & operator /= ( GlobalPoint & a, float b )
  {
    a = GlobalPoint ( a.x() / b, a.y() / b, a.z() / b );
    return a;
  };

  GlobalPoint average ( const vector < MyPair > & pairs, int nq )
  {
    GlobalPoint location(0,0,0);
    for ( vector< MyPair >::const_iterator i=pairs.begin(); i!=( pairs.begin() + nq ); ++i )
      location+=*( i->second );
    location/=nq;
    return location;
  };

  GlobalPoint average ( const vector < MyPairWt > & pairs, int nq )
  {
    GlobalPoint location(0,0,0);
    for ( vector< MyPairWt >::const_iterator i=pairs.begin(); i!=( pairs.begin() + nq ); ++i )
      location+=(i->second)->first;
    location/=nq;
    return location;
  };

  typedef SMS::SMSType SMSType;
};

SMS::SMS ( SMSType tp , float q ) : theType(tp) , theRatio(q) {};


GlobalPoint SMS::location ( const vector<GlobalPoint> & data ) const
{
  if ( theType & Weighted )
  {
    cout << "[SMS] warning: Weighted SMS was asked for, but data are "
         << "weightless!" << endl;
  };
  int nobs=data.size();
  int nq=(int) ceil( theRatio*nobs);
  // cout << "nobs= " << nobs << "  nq= " << nq << endl;

  // Compute distances
  vector<MyPair> pairs;

  for ( vector< GlobalPoint >::const_iterator i=data.begin(); i!=data.end() ; ++i )
  {
    vector < float > D;
    // Compute squared distances to all points
    for ( vector< GlobalPoint >::const_iterator j=data.begin(); j!=data.end() ; ++j )
    { D.push_back ( (*j - *i).mag2() ); }
    // Find q-quantile in each row of the distance matrix
    sort( D.begin(), D.end() );
    MyPair tmp (  D[nq-1], &(*i) );
    pairs.push_back ( tmp );
  };

  // Sort pairs by first element
  sort( pairs.begin(), pairs.end(), Sorter() );
  if ( !(theType & SMS::Interpolate) &&
       !(theType & SMS::Iterate) )
  {
    // we dont interpolate, we dont iterate, so we can stop right here.
    // cout << "No interpolation, no iteration" << endl;
    return *(pairs.begin()->second);
  };

  // we dont iterate, or we dont have anything to iterate (anymore?)
  // so we stop here

  // cout << "nobs= " << nobs << "  nq= " << nq << endl;
  if (!(theType & SMS::Iterate) || nq<=2)
    return average ( pairs, nq );

  // we iterate (recursively)

  vector < GlobalPoint > data1;
  vector<MyPair>::iterator j;

  for ( j=pairs.begin(); j-pairs.begin()<nq; ++j)
     data1.push_back(*(j->second));

  return this->location( data1 );

};


GlobalPoint SMS::location (  const vector < GlPtWt > & wdata ) const
{
  if ( !(theType & Weighted) )
  {
    vector < GlobalPoint > points;
    for ( vector< GlPtWt >::const_iterator i=wdata.begin(); 
          i!=wdata.end() ; ++i )
    {
      points.push_back ( i->first );
    };
    if ( debug() )
    {
      cout << "[SMS] Unweighted SMS was asked for; ignoring the weights."
           << endl;
    };
    return location ( points );
  };
  // int nobs=wdata.size();
  // Sum of weights
  float Sumw=0;
  vector< GlPtWt >::const_iterator i,j;
  for ( i=wdata.begin() ; i!=wdata.end() ; ++i)
    Sumw+=i->second;

  // Compute pairwise distances
  vector <MyPairWt> pairs;
  for ( i=wdata.begin(); i!=wdata.end() ; ++i )
  {
    vector < FloatPair > D;
    // Compute squared distances to all points
    for ( j=wdata.begin(); j!=wdata.end() ; ++j )
      D.push_back ( FloatPair( (j->first - i->first).mag2() , j->second ) ) ;
    // Find weighted q-quantile in the distance vector
    sort( D.begin(), D.end() );
    float sumw=0;
    vector< FloatPair >::const_iterator where;
    for ( where=D.begin(); where!=D.end(); ++where )
    {
      sumw+=where->second;
      // cout << sumw << endl;
      if (sumw>Sumw*theRatio) break;
    }
    MyPairWt tmp ( where->first, &(*i) );
    pairs.push_back ( tmp );
    // cout << where->first << endl;
  };

  // Sort pairs by first element
  sort( pairs.begin(), pairs.end(), Sorter() );

  // Find weighted q-quantile in the list of pairs
  float sumw=0;
  int nq=0;
  vector < MyPairWt >::const_iterator k;
  for (k=pairs.begin(); k!=pairs.end(); ++k )
  {
    sumw+=k->second->second;
    ++nq;
    if (sumw>Sumw*theRatio) break;
  }

  // cout << "nobs= " << nobs << "  nq= " << nq << endl;

  if ( !(theType & SMS::Interpolate) &&
       !(theType & SMS::Iterate) )
  {
    // we dont interpolate, we dont iterate, so we can stop right here.
    // cout << "No interpolation, no iteration" << endl;
    return pairs.begin()->second->first;
  };




  // we dont iterate, or we dont have anything to iterate (anymore?)
  // so we stop here

  // cout << "nobs= " << nobs << "  nq= " << nq << endl;
  if (!(theType & SMS::Iterate) || nq<=2) return average ( pairs, nq );

  // we iterate (recursively)

  vector<GlPtWt> wdata1;

  for ( k=pairs.begin(); k-pairs.begin()<nq; ++k)
    wdata1.push_back(*(k->second));

  return this->location( wdata1 );

};
