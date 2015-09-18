// This implements the CSC gas ionization simulation.
// It requires the GEANT3-generated tables in the input file 
// (default) MuonData/EndcapData/collisions.dat

// Outstanding problems to fix 15-Oct-2003
// - ework must be re-tuned once eion or ework are removed before delta e range estimation.
// - logic of long-range delta e's must be revisited.
// - The code here needs to have diagnostic output removed, or reduced. PARTICULARLY filling of std::vectors!
// - 'gap' is a misnomer, when simhit entry and exit don't coincide with gap edges
//  so the CSCCrossGap might better be named as a simhit-related thing. 
// 22-Jan-2004 Corrected position of trap for 'infinite' loop while
//    generating steps. Output files (debugV-flagged only) require SC flag.
//    Increase deCut from 10 keV to 100 keV to accomodate protons!
// Mar-2015: cout to LogVerbatim. Change superseded debugV-flagged code to flag from config.

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimMuon/CSCDigitizer/src/CSCGasCollisions.h"
#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandFlat.h"

// Only needed if file writing code is activated again
// #include <iostream>
#include <fstream>
#include <cassert>

// stdlib math functions
// for 'abs':
#include <cstdlib>
// for usual math functions:
#include <cmath>

// stdlib container trickery
// for 'for_each':
#include <algorithm>
// for 'greater' and 'less':
#include <functional>
// for 'accumulate':
#include <numeric>
#include <iterator>

using namespace std;

/* Gas mixture is Ar/CO2/CF4 = 40/50/10
We'll use the ionization energies
Ar    15.8 eV
CO2   13.7 eV
CF4    17.8
to arrive at a weighted average of eion = 14.95 */

CSCGasCollisions::CSCGasCollisions( const edm::ParameterSet & pset ) 
 : me("CSCGasCollisions"),
  gasDensity( 2.1416e-03 ), 
  deCut( 1.e05 ), eion( 14.95 ), ework( 34.0 ), clusterExtent( 0.001 ),
  theGammaBins(N_GAMMA, 0.), theEnergyBins(N_ENERGY, 0.), 
  theCollisionTable(N_ENTRIES, 0.), theCrossGap( 0 ),
  theParticleDataTable(0),
  saveGasCollisions_ ( false ), dumpGasCollisions_( false )
{

  dumpGasCollisions_ = pset.getUntrackedParameter<bool>("dumpGasCollisions");

  edm::LogInfo(me) << "Constructing a " << me << ":"; 
  edm::LogInfo(me) << "gas density = " << gasDensity << " g/cm3";
  edm::LogInfo(me) << "max eloss per collision allowed = " << deCut/1000. 
       << " keV (for higher elosses, hits should have been simulated.)";
  edm::LogInfo(me) << "ionization threshold = " << eion << " eV";
  edm::LogInfo(me) << "effective work function = " << ework << " eV";
  edm::LogInfo(me) << "cluster extent = " << clusterExtent*1.e04 << " micrometres";
  edm::LogInfo(me) << "dump gas collision and simhit information? " << dumpGasCollisions();
  edm::LogInfo(me) << "save gas collision information? -NOT YET IMPLEMENTED- " << saveGasCollisions();

  readCollisionTable();
}

CSCGasCollisions::~CSCGasCollisions() {
  edm::LogInfo(me) << "Destructing a " << me;
  delete theCrossGap;
}

void CSCGasCollisions::readCollisionTable() {

  // I'd prefer to allow comments in the data file which means
  // complications of reading line-by-line and then item-by-item.

  // Is float OK? Or do I need double?

  // Do I need any sort of error trapping?

  // We use the default CMSSW data file path
  // We use the default file name 'collisions.dat'
   
  // This can be reset in .orcarc by SimpleConfigurable
  //    Muon:Endcap:CollisionsFile

  string path( getenv( "CMSSW_SEARCH_PATH" ) );
  // TODO make configurable
  string colliFile = "SimMuon/CSCDigitizer/data/collisions.dat";
  FileInPath f1( path, colliFile );
  if (f1() == 0 ) {
    string errorMessage = "Input file " + colliFile + "not found";
    edm::LogError("CSCGasCollisions") << errorMessage << " in path " << path
          << "\nSet Muon:Endcap:CollisionsFile in .orcarc to the "
         " location of the file relative to ORCA_DATA_PATH." ;
    throw cms::Exception( " Endcap Muon gas collisions data file not found.");
  }
  else {
    edm::LogInfo(me) << ": reading " << f1.name();
  }

  ifstream & fin = *f1();

  if (fin.fail()) {
    string errorMessage = "Cannot open input file " + path + colliFile;
    edm::LogError("CSCGasCollisions") << errorMessage;
    throw cms::Exception(errorMessage);
  }

  fin.clear( );                  // Clear eof read status
  fin.seekg( 0, ios::beg );      // Position at start of file

  // @@ We had better have the right sizes everywhere or all
  // hell will break loose. There's no trapping.
  LogTrace(me) << "Reading gamma bins";
  int j = 0;
  for(int i = 0; i<N_GAMMA; ++i) {
    fin >> theGammaBins[i];
    LogTrace(me) << ++j << "   " << theGammaBins[i];
  }

  j = 0;
  LogTrace(me) << "Reading energy bins \n";
  for(int i = 0; i<N_ENERGY; ++i) {
     fin >> theEnergyBins[i];
     LogTrace(me) << ++j << "   " << theEnergyBins[i];
  }

  j = 0;
  LogTrace(me) << "Reading collisions table \n";

  for(int i = 0; i<N_ENTRIES; ++i) {
     fin >> theCollisionTable[i];
     LogTrace(me) << ++j << "   " << theCollisionTable[i];
  }

  fin.close();
}


void CSCGasCollisions::setParticleDataTable(const ParticleDataTable * pdt)
{
  theParticleDataTable = pdt;
}


void CSCGasCollisions::simulate( const PSimHit& simhit, 
   std::vector<LocalPoint>& positions, std::vector<int>& electrons, CLHEP::HepRandomEngine* engine ) {

  const float epsilonL = 0.01;                     // Shortness of simhit 'length'
  //  const float max_gap_z = 1.5;                     // Gas gaps are 0.5 or 1.0 cm

  // Note that what I call the 'gap' may in fact be the 'length' of a PSimHit which
  // does not start and end on the gap edges. This confuses the nomenclature at least.

  double mom    = simhit.pabs();                   // in GeV/c - see MuonSensitiveDetector.cc
  //  int iam       = simhit.particleType();           // PDG type
  delete theCrossGap;                              // before building new one
  assert(theParticleDataTable != 0);
  ParticleData const * particle = theParticleDataTable->particle( simhit.particleType() );
  double mass = 0.105658; // assume a muon
  if(particle == 0)
  {
     edm::LogError("CSCGasCollisions") << "Cannot find particle of type " << simhit.particleType()
            << " in the PDT";
  }
  else 
  {
    mass = particle->mass();
  }

  theCrossGap   = new CSCCrossGap( mass, mom, simhit.exitPoint() - simhit.entryPoint() );
  float gapSize = theCrossGap->length();

  // Test the simhit 'length' (beware of angular effects)
  //  if ( gapSize <= epsilonL || gapSize > max_gap_z ) {
  if ( gapSize <= epsilonL ) {
    edm::LogVerbatim("CSCDigitizer") 
      << "[CSCGasCollisions] WARNING! simhit entry and exit are too close - skipping simhit:"
      << "\n entry = " << simhit.entryPoint()
      << ": exit  = " << simhit.exitPoint()
      << "\n particle type = " << simhit.particleType() << " : momentum = " << simhit.pabs()
      << " GeV/c : energy loss = " << simhit.energyLoss()*1.E06 << " keV"
      << ", gapSize = " << gapSize << " cm (< epsilonL = " << epsilonL << " cm)";
    return; //@@ Just skip this PSimHit
  }

  // Interpolate the table for current gamma value
  // Extract collisions binned by energy loss values, for this gamma
  std::vector<float> collisions(N_ENERGY);
  double loggam = theCrossGap->logGamma(); 
  fillCollisionsForThisGamma( static_cast<float>(loggam), collisions );

  double anmin = exp(collisions[N_ENERGY-1]);
  double anmax = exp(collisions[0]);
  double amu   = anmax - anmin;

  LogTrace(me) << "collisions extremes = " << collisions[N_ENERGY-1]
     << ", " << collisions[0] << "\n"
     << "anmin = " << anmin << ", anmax = " << anmax << "\n"
     << "amu = " << amu << "\n";

  float dedx       = 0.; // total energy loss
  double sum_steps = 0.; // total distance across gap (along simhit direction)
  int n_steps      = 0;  // no. of steps/primary collisions
  int n_try        = 0;  // no. of tries to generate steps
  double step      = -1.; // Sentinel for start

  LocalPoint layerLocalPoint( simhit.entryPoint() );

  // step/primary collision loop
  while ( sum_steps < gapSize) {
    ++n_try;
    if ( n_try > MAX_STEPS ) {
      int maxst = MAX_STEPS;
      edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! n_try = " << n_try 
	   << " is > MAX_STEPS = " << maxst << " - skipping simhit:"
           << "\n entry = " << simhit.entryPoint()
           << ": exit  = " << simhit.exitPoint()
           << "\n particle type = " << simhit.particleType() << " : momentum = " << simhit.pabs()
           << " GeV/c : energy loss = " << simhit.energyLoss()*1.E06 << " keV"
           << "\n gapSize = " << gapSize << " cm, last step = " << step 
           << " cm, sum_steps = " << sum_steps << " cm, n_steps = " << n_steps;
      break;
    }
    step = generateStep( amu, engine );
    if ( sum_steps + step > gapSize ) break; // this step goes too far

    float eloss = generateEnergyLoss( amu, anmin, anmax, collisions, engine );

    // Is the eloss too large? (then GEANT should have produced hits!)
    if ( eloss > deCut ) {
      edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! eloss = " << eloss 
              << " eV is too large (> " << deCut << " eV) - trying another collision";
      continue; // to generate another collision/step
    }

    dedx      += eloss; // the energy lost from the ionizing particle
    sum_steps += step;  // the position of the ionizing particle
    ++n_steps;          // the number of primary collisions
    
    if (n_steps > MAX_STEPS ) { // Extra-careful trap for bizarreness
        edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! " << n_steps 
             << " is too many steps -  skipping simhit:"
             << "\n entry = " << simhit.entryPoint()
             << ": exit  = " << simhit.exitPoint()
             << "\n particle type = " << simhit.particleType() << " : momentum = " << simhit.pabs()
             << " GeV/c : energy loss = " << simhit.energyLoss()*1.E06 << " keV"
	     << "\ngapSize=" << gapSize << " cm, last step=" << step << " cm, sum_steps=" << sum_steps << " cm";
        break;
    }
    LogTrace(me) << "sum_steps = " << sum_steps << " cm , dedx = " << dedx << " eV";
    
    // Generate ionization. 
    // eion is the minimum energy at which ionization can occur in the gas
    if ( eloss > eion  ) {
      layerLocalPoint += step * theCrossGap->unitVector(); // local point where the collision occurs
      ionize( eloss, layerLocalPoint );
    }
    else {
      LogTrace(me) << "Energy available = " << eloss << " eV is too low for ionization (< eion = " << eion << " eV)";
    }

  } // step/collision loop

  if ( dumpGasCollisions() ) writeSummary( n_try, n_steps, sum_steps, dedx, simhit );

  // Return values in two container arguments
  positions = theCrossGap->ionClusters();
  electrons = theCrossGap->electrons();
  return;
}

double CSCGasCollisions::generateStep( double avCollisions, CLHEP::HepRandomEngine* engine ) const
{
// Generate a m.f.p.  (1/avCollisions = cm/collision)
  double step = (CLHEP::RandExponential::shoot(engine))/avCollisions;

// Without using CLHEP: approx random exponential by...
//    double da = double(rand())/double(RAND_MAX);
//    double step = -log(1.-da)/avCollisions;

  LogTrace(me)  << " step = " << step << " cm";
    // Next line only used to fill a container of 'step's for later diagnostic dumps
  if ( dumpGasCollisions() ) theCrossGap->addStep( step );
    return step;
}

float CSCGasCollisions::generateEnergyLoss( double avCollisions, double anmin, double anmax, 
                   const std::vector<float>& collisions, CLHEP::HepRandomEngine* engine ) const
{
// Generate a no. of collisions between collisions[0] and [N_ENERGY-1]
   float lnColl = log(CLHEP::RandFlat::shoot(engine, anmin, anmax));

// Without using CLHEP: approx random between anmin and anmax
//    double ra = double(rand())/double(RAND_MAX)*avCollisions;
//    cout << "ra = " << ra << std::endl;
//    float lnColl = static_cast<float>( log( ra ) );

    // Find energy loss for that number
    float lnE    = lnEnergyLoss( lnColl, collisions );
    float eloss  = exp(lnE);
    // Compensate if gamma was actually below 1.1
    if ( theCrossGap->gamma() < 1.1 ) eloss = eloss * 0.173554/theCrossGap->beta2();
    LogTrace(me) << "eloss = " << eloss << " eV";
    // Next line only used to fill container of eloss's for later diagnostic dumps
    if ( dumpGasCollisions() ) theCrossGap->addEloss( eloss );
    return eloss;
}

void  CSCGasCollisions::ionize( double energyAvailable, LocalPoint startHere ) const
{
  while ( energyAvailable > eion ) {
    LogTrace(me) << "     NEW CLUSTER " << theCrossGap->noOfClusters() + 1 <<
	       " AT " << startHere;
    LocalPoint newCluster( startHere );
    theCrossGap->addCluster(newCluster);

  //@@ I consider NOT subtracting eion before calculating range to be a bug.
  //@@ But this changes tuning of the algorithm so leave it until after the big rush to 7_5_0
  //@@  energyAvailable -= eion; 

  // Sauli CERN 77-09: delta e range with E in MeV (Sauli references Kobetich & Katz 1968
  // but that has nothing to do with this expression! He seems to have  made a mistake.)
    // I found the Sauli-quoted relationship in R. Glocker, Z. Naturforsch. Ba, 129 (1948):
    // delta e range R = aE^n with a=710, n=1.72 for E in MeV and R in mg/cm^2
    // applicable over the range E = 0.001 to 0.3 MeV.

  // Take HALF that range. //@@ Why? Why not...
    double range = 0.5 * (0.71/gasDensity)*pow( energyAvailable*1.E-6, 1.72);
    LogTrace(me) << " range = " << range << " cm";
    if ( range < clusterExtent ) {

      // short-range delta e
          // How many electrons can we make? Now use *average* energy for ionization (not *minimum*)
      int nelec = static_cast<int>(energyAvailable/ework);
      LogTrace(me) << "short-range delta energy in = " << energyAvailable << " eV";
      //energyAvailable -= nelec*(energyAvailable/ework);
      energyAvailable -= nelec*ework;
	    // If still above eion (minimum, not average) add one more e
      if ( energyAvailable > eion ) {
        ++nelec;
        energyAvailable -= eion;
      }
      LogTrace(me) << "short-range delta energy out = " << energyAvailable << " eV, nelec = " << nelec;
      theCrossGap->addElectrons( nelec );
      break;

     }
     else {
      // long-range delta e
         LogTrace(me) << "long-range delta \n"
             << "no. of electrons in cluster now = " << theCrossGap->noOfElectrons();
         theCrossGap->addElectrons( 1 ); // Position is at startHere still

          bool new_range = false;
          while ( !new_range && (energyAvailable>ework) ) {
       	    energyAvailable -= ework;
            while ( energyAvailable > eion ) {
              double range2 = 0.5 * 0.71/gasDensity*pow( 1.E-6*energyAvailable, 1.72);
              double drange = range - range2;
              LogTrace(me) << "  energy left = " << energyAvailable << 
		" eV, range2 = " << range2 << " cm, drange = " << drange << " cm";
              if ( drange < clusterExtent ) {
                theCrossGap->addElectronToBack(); // increment last element
              }
              else {
                startHere += drange*theCrossGap->unitVector(); // update delta e start position
                range = range2;                       // update range
                new_range = true;                     // Test range again
                LogTrace(me) << "reset range to range2 = " << range 
                             << " from startHere = " << startHere << "  and iterate";
              }
       	      break; // out of inner while energyAvailable>eion

            } // inner while energyAvailable>eion

          } // while !new_range && energyAvailable>ework

	  // energyAvailable now less than ework, but still may be over eion...add an e
          if ( energyAvailable > eion ) {
            energyAvailable -= ework; // yes, it may go negative
            theCrossGap->addElectronToBack(); // add one more e
          }

        } // if range

      } // outer while energyAvailable>eion
}

void CSCGasCollisions::writeSummary( int n_try, int n_steps, double sum_steps, float dedx, 
           const PSimHit& simhit ) const
{
  // Switched from std::cout to LogVerbatim, Mar 2015

  std::vector<LocalPoint> ion_clusters   = theCrossGap->ionClusters();
  std::vector<int> electrons             = theCrossGap->electrons();
  std::vector<float> elosses             = theCrossGap->eLossPerStep();
  std::vector<double> steps              = theCrossGap->stepLengths();

  edm::LogVerbatim("CSCDigitizer") << "------------------"
                                 << "\nAFTER CROSSING GAP";
  /*
  edm::LogVerbatim("CSCDigitizer") << "no. of steps tried             = " << n_try
                                 << "\nno. of steps from theCrossGap  = " << theCrossGap->noOfSteps()
				 << "\nsize of steps vector           = " << steps.size();
                                 
  edm::LogVerbatim("CSCDigitizer") << "no. of collisions (steps)      = " << n_steps
			         << "\nsize of elosses vector         = " << elosses.size()
				 << "\nsize of ion clusters vector    = " << ion_clusters.size()
                                 << "\nsize of electrons vector       = " << electrons.size();
  */

  size_t nsteps = n_steps; // force ridiculous type conversion
  size_t mstep = steps.size() - 1; // final step gets filled but is outside gas gap - unless we reach MAX_STEPS
  if ( (nsteps != MAX_STEPS) && (nsteps != mstep) ) {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! no. of steps = " << nsteps 
                                     << " .ne. steps.size()-1 = " << mstep;
  }
  size_t meloss = elosses.size();

  if ( nsteps != meloss ){
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! no. of steps = " << nsteps 
                                     << " .ne. no. of elosses = " << meloss;
  }
  else {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] # / length of step / energy loss per collision:";
    for ( size_t i = 0; i != nsteps; ++i ) {
      edm::LogVerbatim("CSCDigitizer") << i+1 << " / S: " << steps[i] << " / E: " << elosses[i];
    }
  }

  size_t mclus = ion_clusters.size();
  size_t melec = electrons.size();

  if ( mclus != melec ){
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] WARNING! size of cluster vector = " << mclus 
            << " .ne. size of electrons vector = " << melec;
  }
  else {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] # / postion of cluster / electrons per cluster: ";
    for ( size_t i = 0; i != mclus; ++i ) {
      edm::LogVerbatim("CSCDigitizer") << i+1 <<  " / I: " << ion_clusters[i] << " / E: " << electrons[i];
    }
  }

  int n_ic = count_if( elosses.begin(), elosses.end(), bind2nd(greater<float>(), eion) );

  edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] total no. of collision steps = " << n_steps;
  edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] total sum of steps = " << sum_steps << " cm";
  if ( nsteps > 0 ) edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] average step length = " << 
      sum_steps/float(nsteps) << " cm";
  edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] total energy loss across gap = " << 
      dedx << " eV = " << dedx/1000. << " keV";
  edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] no. of primary ionizing collisions across gap = no. with eloss > eion = "
      << eion << " eV = " << n_ic;
  if ( nsteps > 0 ) edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] average energy loss/collision = " <<  
      dedx/float(nsteps) << " eV";

  std::vector<int>::const_iterator bigger = find(electrons.begin(), electrons.end(), 0 );
  if ( bigger != electrons.end() ) {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] TROUBLE! There is a cluster with 0 electrons.";
  }

  int n_e = accumulate(electrons.begin(), electrons.end(), 0 );

  edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] SUMMARY: simhit"
      << "\n entry = " << simhit.entryPoint()
      << ": exit  = " << simhit.exitPoint()
      << "\n particle type = " << simhit.particleType() << " : momentum = " << simhit.pabs()
      << " GeV/c : energy loss = " << simhit.energyLoss()*1.E06 << " keV";

  if ( nsteps > 0 ) {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisions] SUMMARY: ionization" 
      << " : steps= " << nsteps << " : sum(steps)= " << sum_steps << " cm : <step>= " 
      << sum_steps/float(nsteps) << " cm"
      << " : ionizing= " << n_ic << " : ionclus= " << mclus << " : total e= " << n_e
      << " : <dedx/step>= " << dedx/float(nsteps)
      << " eV : <e/ionclus>= " << float(n_e)/float(mclus)
      << " : dedx= " << dedx/1000. << " keV";
  }
  else {
    edm::LogVerbatim("CSCDigitizer") << "[CSCGasCollisons] ERROR? no collision steps!";
  }

  // Turn off output file -  used for initial development
  //  if ( saveGasCollisions() ) {
  //    ofstream of0("osteplen.dat",ios::app);
  //    std::copy( steps.begin(), steps.end(), std::ostream_iterator<float>(of0,"\n"));

  //    ofstream of1("olperc.dat",ios::app);
  //    std::copy( elosses.begin(), elosses.end(), std::ostream_iterator<float>(of1,"\n"));

  //   ofstream of2("oclpos.dat",ios::app);
  //   std::copy( ion_clusters.begin(), ion_clusters.end(), std::ostream_iterator<LocalPoint>(of2,"\n"));

  //   ofstream of3("oepercl.dat",ios::app);
  //   std::copy( electrons.begin(), electrons.end(), std::ostream_iterator<int>(of3,"\n"));
  // }

}

float CSCGasCollisions::lnEnergyLoss( float lnCollisions, 
   const std::vector<float>& collisions ) const {

  float lnE = -1.;

   // Find collision[] bin in which lnCollisions falls
    std::vector<float>::const_iterator it = find(collisions.begin(),
      collisions.end(), lnCollisions );

    if ( it != collisions.end() ) {
      // found the value
      std::vector<float>::difference_type ihi = it - collisions.begin();
      LogTrace(me) << ": using one energy bin " << ihi << " = " 
                         << theEnergyBins[ihi]
		         << " for lnCollisions = " << lnCollisions;
      lnE = theEnergyBins[ihi];
    }
    else {
      // interpolate the value
      std::vector<float>::const_iterator loside = find_if(collisions.begin(),
        collisions.end(), bind2nd(less<float>(), lnCollisions));
      std::vector<float>::difference_type ilo = loside - collisions.begin();
      if ( ilo > 0 ) {
        LogTrace(me) << ": using energy bin " 
                           << ilo-1 << " and " << ilo;
        lnE = theEnergyBins[ilo-1] + (lnCollisions-collisions[ilo-1])*
          (theEnergyBins[ilo]-theEnergyBins[ilo-1]) /
             (collisions[ilo]-collisions[ilo-1]);
      }
      else {
        LogTrace(me) << ": using one energy bin 0 = " 
                           << theEnergyBins[0]
			   << " for lnCollisions = " << lnCollisions;
        lnE = theEnergyBins[0]; //@@ WHAT ELSE TO DO?
      }
    }

    return lnE;
}

void CSCGasCollisions::fillCollisionsForThisGamma( float logGamma,
   std::vector<float>& collisions ) const
{
  std::vector<float>::const_iterator bigger = find_if(theGammaBins.begin(),
    theGammaBins.end(), bind2nd(greater<float>(), logGamma));

  if ( bigger == theGammaBins.end() ) {
    // use highest bin
    LogTrace(me) << ": using highest gamma bin" 
		      << " for logGamma = " << logGamma;
    for (int i=0; i<N_ENERGY; ++i)
      collisions[i] = theCollisionTable[i*N_GAMMA];
  }
  else {
    // use bigger and its lower neighbour
    std::vector<float>::difference_type ihi = bigger - theGammaBins.begin();
    if ( ihi > 0 ) {
      double dlg2 = *bigger--; // and decrement after deref
      //LogTrace(me) << ": using gamma bins " 
      //                   << ihi-1 << " and " << ihi;
      double dlg1 = *bigger;   // now the preceding element
      double dlg = (logGamma-dlg1)/(dlg2-dlg1);
      double omdlg = 1. - dlg;
      for (int i=0; i<N_ENERGY; ++i)
        collisions[i] = theCollisionTable[i*N_GAMMA+ihi-1]*omdlg +
          theCollisionTable[i*N_GAMMA+ihi]*dlg;
    }
    else {
      // bigger has no lower neighbour
       LogTrace(me) << ": using lowest gamma bin" 
       		         << " for logGamma = " << logGamma;

      for (int i=0; i<N_ENERGY; ++i)
        collisions[i] = theCollisionTable[i*N_GAMMA];
    }
  }
}
