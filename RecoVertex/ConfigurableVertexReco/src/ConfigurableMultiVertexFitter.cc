#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableMultiVertexFitter.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
// #include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableTrimmedKalmanFinder.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexBSeeder.h"

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
    ret.addParameter<double>("sigmacut",9.0);
    ret.addParameter<double>("Tini",8.0);
    ret.addParameter<double>("ratio",0.25);
    ret.addParameter<int>("cheat",0);
    edm::ParameterSet nest;
    nest.addParameter<string>("finder","mbs");
    ret.addParameter<edm::ParameterSet>("ini",nest);
    return ret;
  }

  const AnnealingSchedule * schedule ( const edm::ParameterSet & m )
  {
    return new GeometricAnnealing(
        m.getParameter<double>("sigmacut"), 
        m.getParameter<double>("Tini"),
        m.getParameter<double>("ratio") );
  }
  
  const VertexReconstructor * initialiser ( const edm::ParameterSet & p )
  {
    // cout << "[ConfigurableMultiVertexFitter] ini: " << p << endl;
    return new ConfigurableVertexReconstructor ( p );
  } 
}

ConfigurableMultiVertexFitter::ConfigurableMultiVertexFitter() :
  theRector ( new MultiVertexReconstructor( MultiVertexBSeeder() ) ),
  theCheater(0)
{}

void ConfigurableMultiVertexFitter::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m = mydefaults();
  m.augment ( n );
  // print ( m );
  const AnnealingSchedule * ann = schedule ( m );
  const VertexReconstructor * ini = initialiser ( m.getParameter<edm::ParameterSet>("ini") );
  if ( theRector ) delete theRector;
  theRector = new MultiVertexReconstructor( *ini, *ann );
  theCheater=m.getParameter<int>("cheat");
  delete ann;
  delete ini;
}

ConfigurableMultiVertexFitter::~ConfigurableMultiVertexFitter()
{
  if ( theRector ) delete theRector;
}

ConfigurableMultiVertexFitter::ConfigurableMultiVertexFitter 
    ( const ConfigurableMultiVertexFitter & o ) :
  theRector ( o.theRector->clone() ),
  theCheater(o.theCheater)
{}


ConfigurableMultiVertexFitter * ConfigurableMultiVertexFitter::clone() const
{
  return new ConfigurableMultiVertexFitter ( *this );
}

vector < TransientVertex > ConfigurableMultiVertexFitter::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  if ( theCheater==1 && t.size()>3 )
  {
    std::vector < reco::TransientTrack > primaries;
    reco::TransientTrack p1=t[0];
    reco::TransientTrack p2=t[1];
    reco::TransientTrack p3=t[2];
    reco::TransientTrack p4=t[3];
    primaries.push_back ( p1 );
    primaries.push_back ( p2 );
    primaries.push_back ( p3 );
    primaries.push_back ( p4 );
    /*
    cout << "[ConfigurableMultiVertexFitter] primaries: ";
    for ( vector< reco::TransientTrack >::const_iterator i=primaries.begin(); 
          i!=primaries.end() ; ++i )
    {
      cout << i->id() << "  ";
    }
    cout << endl;
    */
    return theRector->vertices ( t, primaries );
  }

  // test code ends
  // 
  return theRector->vertices ( t );
}

edm::ParameterSet ConfigurableMultiVertexFitter::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableMultiVertexFitter > t ( "mvf", "Multi Vertex Fitter" );
}
