/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Nov                            ///
///                                      ///
/// ////////////////////////////////////////

#ifndef BEAM_BUILDER_H
#define BEAM_BUILDER_H

#include <memory>
#include <map>
#include <vector>

/// WARNING NP** davvero ci servono tutti questi include?
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

/** ************************ **/
/**                          **/
/**   DECLARATION OF CLASS   **/
/**                          **/
/** ************************ **/

class L1TkBeamBuilder : public edm::EDProducer {

  public:

  private:
    /// Data members

    /// Gaussian
    bool isGauss;
    double MeanX;
    double MeanY;
    double MeanZ;
    double SigmaX;
    double SigmaY;
    double SigmaZ;
    //double TimeOffset;

    /// Flat
    bool isFlat;
    double MinX;
    double MinY;
    double MinZ;
    double MaxX;
    double MaxY;
    double MaxZ;
    //double TimeOffset;

    /// Beta Function
    bool isBeta;
    double Phi;
    double BetaStar;
    double Emittance;
    double Alpha;
    //double SigmaZ;
    //double TimeOffset;
    double X0;
    double Y0;
    double Z0;

    /// Other stuff
    //const cmsUpgrades::classInfo *mClassInfo;

  public:
    /// Constructors
    explicit L1TkBeamBuilder( const edm::ParameterSet& iConfig );
    /// Destructor
    ~L1TkBeamBuilder();

  private:
    /// ///////////////// ///
    /// MANDATORY METHODS ///
    virtual void beginRun( edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    /// ///////////////// ///
    /// AUXILIARY METHODS ///

}; /// Close class



/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

/// ////////////////////////// ///
/// CONSTRUCTORS & DESTRUCTORS ///
/// ////////////////////////// ///

/// Constructors
/// Default is for SimHits
L1TkBeamBuilder::L1TkBeamBuilder( const edm::ParameterSet& iConfig )
{
  /// Gaussian
  isGauss = false;
  MeanX = 0.0;
  MeanY = 0.0;
  MeanZ = 0.0;
  SigmaX = 0.0;
  SigmaY = 0.0;
  SigmaZ = 0.0;
  //TimeOffset;

  /// Flat
  isFlat = false;
  MinX = 0.0;
  MinY = 0.0;
  MinZ = 0.0;
  MaxX = 0.0;
  MaxY = 0.0;
  MaxZ = 0.0;
  //TimeOffset;

  /// Beta Function
  isBeta = false;
  Phi = 0.0;
  BetaStar = 0.0;
  Emittance = 0.0;
  Alpha = 0.0;
  //SigmaZ;
  //TimeOffset;
  X0 = 0.0;
  Y0 = 0.0;
  Z0 = 0.0;

  /// Check if it is Gaussian
  if ( iConfig.exists("MeanX") &&
       iConfig.exists("MeanY") &&
       iConfig.exists("MeanZ") &&
       iConfig.exists("SigmaX") &&
       iConfig.exists("SigmaY") &&
       iConfig.exists("SigmaZ") ) {
    isGauss = true;
    MeanX = iConfig.getParameter< double >("MeanX");
    MeanY = iConfig.getParameter< double >("MeanY");
    MeanZ = iConfig.getParameter< double >("MeanZ");
    SigmaX = iConfig.getParameter< double >("SigmaX");
    SigmaY = iConfig.getParameter< double >("SigmaY");
    SigmaZ = iConfig.getParameter< double >("SigmaZ");
  }

  /// Check if it is Flat
  if ( iConfig.exists("MinX") &&
       iConfig.exists("MinY") &&
       iConfig.exists("MinZ") &&
       iConfig.exists("MaxX") &&
       iConfig.exists("MaxY") &&
       iConfig.exists("MaxZ") ) {
    isFlat = true;
    MinX = iConfig.getParameter< double >("MinX");
    MinY = iConfig.getParameter< double >("MinY");
    MinZ = iConfig.getParameter< double >("MinZ");
    MaxX = iConfig.getParameter< double >("MaxX");
    MaxY = iConfig.getParameter< double >("MaxY");
    MaxZ = iConfig.getParameter< double >("MaxZ");
  }

  /// Check if it is Beta
  if ( iConfig.exists("X0") &&
       iConfig.exists("Y0") &&
       iConfig.exists("Z0") &&
       iConfig.exists("Phi") &&
       iConfig.exists("BetaStar") &&
       iConfig.exists("Emittance") &&
       iConfig.exists("Alpha") &&
       iConfig.exists("SigmaZ") ) {
    isBeta = true;
    X0 = iConfig.getParameter< double >("X0");
    Y0 = iConfig.getParameter< double >("Y0");
    Z0 = iConfig.getParameter< double >("Z0");
    Phi = iConfig.getParameter< double >("Phi");
    BetaStar = iConfig.getParameter< double >("BetaStar");
    Emittance = iConfig.getParameter< double >("Emittance");
    Alpha = iConfig.getParameter< double >("Alpha");
    SigmaZ = iConfig.getParameter< double >("SigmaZ");
  }

  produces< std::vector< cmsUpgrades::L1TkBeam > >();
  /// This produces a vector with 1 element, but this way it runs
  /// and any attempt to store something different such as the L1TkBeam
  /// itself... well, crash boom bang.
}

/// Destructor
L1TkBeamBuilder::~L1TkBeamBuilder()
{
  /// Nothing to be done
}



/// ///////////////// ///
/// MANDATORY METHODS ///
/// ///////////////// ///

/// Begin run
void L1TkBeamBuilder::beginRun( edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Print some information when loaded
  std::cout << "L1TkBeamBuilder" << std::endl;
}

/// End run
void L1TkBeamBuilder::endRun( edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Nothing to be done
}

/// Implement the producer
void L1TkBeamBuilder::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  std::auto_ptr< std::vector<cmsUpgrades::L1TkBeam> > BeamForOutput( new std::vector<cmsUpgrades::L1TkBeam> );
  cmsUpgrades::L1TkBeam tempBeam;

  if ( isBeta ) {
    tempBeam.setBeamType( 0 );
    tempBeam.setBeamPosition( X0, Y0, Z0 );
    double SigmaT = sqrt( Emittance*BetaStar );
    tempBeam.setBeamSize( SigmaT, SigmaT, SigmaZ );
  }
  else if ( isGauss ) {
    tempBeam.setBeamType( 1 );
    tempBeam.setBeamPosition( MeanX, MeanY, MeanZ );
    tempBeam.setBeamSize( SigmaX, SigmaY, SigmaZ );
  }
  else if ( isFlat ) {
    tempBeam.setBeamType( 2 );
    double x0 = 0.5*( MaxX + MinX );
    double y0 = 0.5*( MaxY + MinY );
    double z0 = 0.5*( MaxZ + MinZ );
    tempBeam.setBeamPosition( x0, y0, z0 );
    double xw = fabs( MaxX - MinX );
    double yw = fabs( MaxY - MinY );
    double zw = fabs( MaxZ - MinZ );
    tempBeam.setBeamSize( xw, yw, zw );
  }
  BeamForOutput->push_back( tempBeam );

  /// Store it away!
  if ( isBeta || isGauss || isFlat )
    iEvent.put( BeamForOutput );
}



/// ///////////////// ///
/// AUXILIARY METHODS ///
/// ///////////////// ///


#endif


