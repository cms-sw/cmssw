#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAnnealing.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
// #include "RecoVertex/VertexTools/interface/CriticalAnnealing.h"
// #include "RecoVertex/VertexTools/interface/EquityAnnealing.h"
#include <string>
#include <iostream>

using namespace std;

ConfigurableAnnealing::ConfigurableAnnealing ( const edm::ParameterSet & m ) :
  theImpl( 0 )
{
  string type = m.getParameter<string>("annealing");
  if ( type == "crit" )
  {
    cout << "[ConfigurableAnnealing] no critical annealing!" << endl;
	exit(-1);
	  /*
    theImpl = new CriticalAnnealing(
                    m.getParameter<double>("sigmacut"),
                    m.getParameter<double>("Tini"),
                    m.getParameter<double>("ratio") );
					*/
  } else if ( type == "crit" ) {
    cout << "[ConfigurableAnnealing] no equity annealing!" << endl;
	exit(-1);
	  /*
    theImpl = new EquityAnnealing(
                    m.getParameter<double>("sigmacut"),
                    m.getParameter<double>("Tini"),
                    m.getParameter<double>("ratio") );
					*/
  } else {
    theImpl = new GeometricAnnealing(
                    m.getParameter<double>("sigmacut"),
                    m.getParameter<double>("Tini"),
                    m.getParameter<double>("ratio") );
  };
}

ConfigurableAnnealing::ConfigurableAnnealing ( const ConfigurableAnnealing & o ) :
  theImpl ( o.theImpl->clone() )
{}

ConfigurableAnnealing * ConfigurableAnnealing::clone() const
{
  return new ConfigurableAnnealing ( *this );
}

ConfigurableAnnealing::~ConfigurableAnnealing()
{
  delete theImpl;
}

void ConfigurableAnnealing::debug() const
{
  theImpl->debug();
}

void ConfigurableAnnealing::anneal()
{
  theImpl->anneal();
}

double ConfigurableAnnealing::weight ( double chi2 ) const
{
  return theImpl->weight ( chi2 );
}

void ConfigurableAnnealing::resetAnnealing()
{
  theImpl->resetAnnealing();
}

inline double ConfigurableAnnealing::phi( double chi2 ) const
{
  return theImpl->phi ( chi2 );
}

double ConfigurableAnnealing::cutoff() const
{
  return theImpl->cutoff();
}

double ConfigurableAnnealing::currentTemp() const
{
  return theImpl->currentTemp();
}

double ConfigurableAnnealing::initialTemp() const
{
  return theImpl->initialTemp();
}

bool ConfigurableAnnealing::isAnnealed() const
{
  return theImpl->isAnnealed();
}
