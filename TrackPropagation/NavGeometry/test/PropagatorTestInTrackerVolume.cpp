
#include "TrackPropagation/NavGeometry/test/stubs/HelixPropagationTestGenerator.h"
///#include "CommonReco/PatternTestTools/interface/StraightLinePropagationTestGenerator.h"
#include "TrackPropagation/NavGeometry/test/stubs/PropagatorTestTree.h"
#include "TrackPropagation/NavGeometry/test/stubs/RandomCylinderGenerator.h"
#include "TrackPropagation/NavGeometry/test/stubs/RandomPlaneGeneratorByAxis.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"

///#include "CommonReco/GeomPropagators/test/OldGtfPropagator.cc"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
//#include "CommonReco/MaterialEffects/interface/PropagatorWithMaterial.h"
//#include "CommonDet/PatternPrimitives/interface/MediumProperties.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"

#include <iostream>
#include <iomanip>

class ConstantMagneticField4T : public MagneticField {
public:
  GlobalVector inTesla( const GlobalPoint& gp) const {return GlobalVector(0,0,4.);}
};

class ConstantMagneticFieldProvider4T : public MagneticFieldProvider<float> {
public:
  virtual LocalVectorType valueInTesla( const LocalPointType& p) const {return LocalVectorType(0,0,4.);}
};

class ConstantMagVolume4T : public MagVolume {
public:
  ConstantMagVolume4T( const PositionType& pos, const RotationType& rot, 
		       DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume( pos, rot, shape, mfp) {}
 
  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const {return true;}

  /// Access to volume faces
  virtual std::vector<VolumeSide> faces() const {return std::vector<VolumeSide>();}
};


using namespace std;

//#include "MagneticField/BaseMagneticField/interface/CMSMagneticFieldFactory.h"

//#include "MagneticField/BaseMagneticField/interface/CMSMagneticFieldLoader.h"

//#include "Utilities/GenUtil/interface/CMSexception.h"
//static CMSMagneticFieldSimpleFactory<CMSMagneticFieldLoader> localFactory;

//
// Compare two propagators by generating random helices. Step forward up
// to the boundaries of the tracker volume and back to the origin. A
// plane, smeared around the local normal plane to the helix, is created
// at each step and trajectory states propagated to it.
//
int main (int argc, char* argv[]) {
  // Capri::Init cinit(argc,argv);

  typedef long double ExtendedDouble;
  //   try {
  //     ((CMSMagneticFieldLoader*)(localFactory.singleton()))->load();
  //   }
  //   catch (Genexception & cexp) {
  //     cout << cexp.what() << endl;
  //     return 8;
  //   }
  MagneticField* theField = new ConstantMagneticField4T;

  //
  // nr. of events, maximum step length, maximum tilt of planes,
  // limits of the volume (in r and z)
  //
  //   SimpleConfigurable<int> maxEvents_conf(10000,"PropagatorTestInTrackerVolume:maxEvents");
  //   int theMaxEvents = maxEvents_conf.value();
  int theMaxEvents = 100;

  //   SimpleConfigurable<float> maxStep_conf(50.,"PropagatorTestInTrackerVolume:maxStep");
  //   float theMaxStep = maxStep_conf.value();
  float theMaxStep = 50.;

  //   SimpleConfigurable<float> maxTilt_conf(0.5,"PropagatorTestInTrackerVolume:maxTilt");
  //   float theMaxTilt = maxTilt_conf.value();
  float theMaxTilt = 0.5;

  //   SimpleConfigurable<float> maxR_conf(120.,"PropagatorTestInTrackerVolume:maxRadius");
  //   float theMaxR = maxR_conf.value();
  float theMaxR = 120.;

  //   SimpleConfigurable<float> maxZ_conf(300.,"PropagatorTestInTrackerVolume:maxZ");
  //   float theMaxZ = maxZ_conf.value();
  float theMaxZ = 300.;

  //   SimpleConfigurable<string> planeType_conf("","PropagatorTestInTrackerVolume:planeType");
  //   if ( planeType_conf.value()!="" ) 
  //     cout << "Configurable PropagatorTestInTrackerVolume:planeType is obsolete !!!" << endl;

  //   SimpleConfigurable<string> 
  //     surfaceType_conf("arbitrary","PropagatorTestInTrackerVolume:surfaceType");
  //   string theSurfaceType = surfaceType_conf.value();
  string theSurfaceType = "arbitrary";

  //   SimpleConfigurable<bool> useHelix_conf(1,"PropagatorTestInTrackerVolume:useHelix");
  //   bool theUseHelix = useHelix_conf.value();
  bool theUseHelix = true;

  //   SimpleConfigurable<bool> 
  //     anyDirection_conf(false,"PropagatorTestInTrackerVolume:anyDirection");
  //   bool theAnyDirection = anyDirection_conf.value();
  bool theAnyDirection = false;

  //   SimpleConfigurable<bool> 
  //     errorPropagation_conf(false,"PropagatorTestInTrackerVolume:propagateErrors");
  //   bool theErrorPropagation = errorPropagation_conf.value();
  bool theErrorPropagation = false;

  //   SimpleConfigurable<bool> 
  //     materialEffects_conf(false,"PropagatorTestInTrackerVolume:materialEffects");
  //   bool theMaterialEffects = materialEffects_conf.value();

  //   SimpleConfigurable<float> 
  //     radLength_conf(0.01,"PropagatorTestInTrackerVolume:radLen");
  //   float theRadLength = radLength_conf.value();
  float theRadLength = 0.01;

  //   SimpleConfigurable<float> xi_conf(0.01,"PropagatorTestInTrackerVolume:xi");
  //   float theXi = xi_conf.value();
  float theXi = 0.01;

  //   if ( theMaterialEffects )
  //     cout << " Rad.Length / Xi = " << theRadLength << "/" << theXi << endl;
  //
  // Timing
  //
  //
  // Histogram class and propagators
  //
  PropagatorTestTree theHistogrammer;

  ConstantMagneticFieldProvider4T theProvider;
  ConstantMagVolume4T theMagVolume( MagVolume::PositionType(0,0,0), MagVolume::RotationType(),
				    ddshapeless, &theProvider);

  Propagator* oldPropagator = new RKPropagatorInS( theMagVolume, alongMomentum);

  Propagator* newPropagator;
  //   if ( theMaterialEffects ) 
  //     newPropagator = new PropagatorWithMaterial(alongMomentum);
  //   else
  newPropagator = new AnalyticalPropagator(theField, alongMomentum);
  if ( theAnyDirection )
    newPropagator->setPropagationDirection(anyDirection);
  //
  // helix generator (values now defined via .orcarc)
  //
  PropagationTestGenerator* trajectoryGenerator;
  ///if ( theUseHelix )
  trajectoryGenerator = new HelixPropagationTestGenerator(theField);
  ///  else
  ///trajectoryGenerator = new StraightLinePropagationTestGenerator();
  //
  // random plane generator
  //
  RandomCylinderGenerator* cylinderGenerator(0);
  RandomPlaneGeneratorByAxis* planeGenerator(0);
  bool surfaceIsPlane;
  if ( theSurfaceType=="cylinder" ) {
    cylinderGenerator = new RandomCylinderGenerator(theMaxZ);
    surfaceIsPlane = false;
  }
  else {
    planeGenerator = new RandomPlaneGeneratorByAxis(theMaxTilt);
    if ( theSurfaceType=="forward" )
      planeGenerator->setType(RandomPlaneGeneratorByAxis::forward);
    else if ( theSurfaceType=="strictForward" )
      planeGenerator->setType(RandomPlaneGeneratorByAxis::strictForward);
    else if ( theSurfaceType=="barrel" )
      planeGenerator->setType(RandomPlaneGeneratorByAxis::barrel);
    else if ( theSurfaceType=="strictBarrel" )
      planeGenerator->setType(RandomPlaneGeneratorByAxis::strictBarrel);
    else
      planeGenerator->setType(RandomPlaneGeneratorByAxis::arbitrary);
    surfaceIsPlane = true;
  }

  for ( int ievt=0; ievt<theMaxEvents; ievt++ ) {
    //
    // generate helix, get starting point and define plane
    //
    ExtendedDouble stot;
    ExtendedDouble stot0(0.);
    trajectoryGenerator->generateStartValues();
    GlobalPoint xStart = trajectoryGenerator->position();
    GlobalVector pStart = trajectoryGenerator->momentum();
    if ( !surfaceIsPlane ) {
      // for cylinders: do one first step (cylindrical geometry
      // is ill defined at origin)
      stot0 = trajectoryGenerator->randomStepForward(theMaxStep);
      xStart = trajectoryGenerator->position();
      pStart = trajectoryGenerator->momentum();
      // check, if still inside volume
      if ( xStart.perp()>theMaxR || fabs(xStart.z())>theMaxZ )  continue;
    }
    cout << " starting at x = " << xStart << endl;
    cout << "             p = " << pStart << endl;
    //      RandomPlaneGenerator::PlanePtr startPlane = (*surfaceGenerator)(xStart,pStart);
    RandomPlaneGenerator::PlanePtr startPlane(0);
    BoundCylinder::BoundCylinderPointer startCylinder(0);
    if ( surfaceIsPlane )
      startPlane = (*planeGenerator)(xStart,pStart);
    else
      startCylinder = (*cylinderGenerator)(xStart,pStart);
    //     if ( theMaterialEffects && theSurfaceType!="cylinder" ) {
    //       startPlane->setMediumProperties(new MediumProperties(theRadLength,theXi));
    //     }
    if ( surfaceIsPlane ) {
      cout << "generated Plane at " << startPlane->position() << endl;
      cout << startPlane->rotation() << endl;
    }
    else {
      cout << "generated Cylinder at r = " << startCylinder->radius() << endl;
    }
    // fill helix parameters into ntuple
    GlobalPoint center(0.,0.,0.);
    if ( theUseHelix ) {
      HelixPropagationTestGenerator* helixGenerator = 
	dynamic_cast<HelixPropagationTestGenerator*>(trajectoryGenerator);
      center = helixGenerator->center();
    }
    theHistogrammer.fillHelix(xStart,pStart,center,
			      trajectoryGenerator->transverseCurvature(),
			      trajectoryGenerator->charge());
    //
    // define trajectory state at start
    //
    GlobalTrajectoryParameters aux(xStart,pStart,trajectoryGenerator->charge(),theField);
    TrajectoryStateOnSurface simState;
    if ( theErrorPropagation ) {
      AlgebraicSymMatrix matrix(5,0);
      // 1% relative on 1/p, 1mRad on theta/phi, 100um on x/y
      matrix[0][0] = 0.01/pStart.mag();
      matrix[1][1] = matrix[2][2] = 0.001;
      matrix[3][3] = matrix[4][4] = 100.e-4;
      if ( surfaceIsPlane )
	simState = TrajectoryStateOnSurface(aux,
					    CurvilinearTrajectoryError(matrix),
					    *startPlane);
      else
	simState = TrajectoryStateOnSurface(aux,
					    CurvilinearTrajectoryError(matrix),
					    *startCylinder);
    }
    else {
      if ( surfaceIsPlane )
	simState = TrajectoryStateOnSurface(aux,*startPlane);
      else
	simState = TrajectoryStateOnSurface(aux,*startCylinder);
    }
    TrajectoryStateOnSurface tsosOld,tsosNew;
    tsosOld = tsosNew = simState;
    //      if ( tsosOld.hasError() )
    //        cout << "Errors at start = " << tsosOld.localError().matrix() << endl;
    //
    // First part: forward propagation towards limits of the tracker volume
    //
    oldPropagator->setPropagationDirection(alongMomentum);
    if ( !theAnyDirection )  newPropagator->setPropagationDirection(alongMomentum);
    GlobalPoint xGen;
    GlobalVector pGen;
    do {
      //
      // make one step and get "true" position / direction and create plane
      //
      stot = trajectoryGenerator->randomStepForward(theMaxStep) - stot0;
      xGen = trajectoryGenerator->position();
      pGen = trajectoryGenerator->momentum();
      //        cout << " stepped to x = " << xGen << endl;
      //        cout << "            p = " << pGen << endl;
      //        RandomPlaneGenerator::PlanePtr currentPlane = (*surfaceGenerator)(xGen,pGen);
      RandomPlaneGenerator::PlanePtr currentPlane(0);
      BoundCylinder::BoundCylinderPointer currentCylinder(0);
      if ( surfaceIsPlane )
	currentPlane = (*planeGenerator)(xGen,pGen);
      else
	currentCylinder = (*cylinderGenerator)(xGen,pGen);
      //       if ( theMaterialEffects && theSurfaceType!="cylinder" ) {
      // 	currentPlane->setMediumProperties(new MediumProperties(theRadLength,theXi));
      //       }
      // continue propagation of states by propagator 1
      if ( tsosOld.isValid() ) {
	if ( surfaceIsPlane )
	  tsosOld = oldPropagator->propagate(tsosOld,*currentPlane);
	else
	  tsosOld = oldPropagator->propagate(tsosOld,*currentCylinder);
	if ( !tsosOld.isValid() )  cout << "  tsosOld is invalid" << endl;
      }
      // continue propagation of states by propagator 2
      if ( tsosNew.isValid() ) {
	if ( surfaceIsPlane )
	  tsosNew = newPropagator->propagate(tsosNew,*currentPlane);
	else
	  tsosNew = newPropagator->propagate(tsosNew,*currentCylinder);
	if ( !tsosNew.isValid() )  cout << "  tsosNew is invalid" << endl;
	// 	else  cout << tsosNew.localParameters().position() << " "
	// 		   << tsosNew.localParameters().momentum() << " "
	// 		   << tsosNew.localParameters().charge() << " "
	// 		   << tsosNew.localParameters().pzSign() << endl;
      }
      // add results to ntuple
      if ( surfaceIsPlane )
	theHistogrammer.addStep(stot,xGen,pGen,
				currentPlane->position(),
				currentPlane->toGlobal(LocalVector(0.,0.,1.)),
				tsosOld,tsosNew);
      else {
	GlobalPoint pos = currentCylinder->position();
	float radius = currentCylinder->radius();
	theHistogrammer.addStep(stot,xGen,pGen,
				GlobalPoint(pos.x(),pos.y(),radius),
				GlobalVector(xGen.x(),xGen.y(),0.).unit(),
				tsosOld,tsosNew);
      }
      // terminate loop if current point is outside volume
    } while ( xGen.perp()<theMaxR && fabs(xGen.z())<theMaxZ );
    //      // errors
    //      if ( tsosOld.isValid() ) {
    //        cout << "Old state outside  = " << tsosOld.localPosition() 
    //             << " " << tsosOld.localMomentum() << endl;
    //        if ( tsosOld.hasError() )
    //  	cout << "Old errors at outside = " << tsosOld.localError().matrix() << endl;
    //      }
    //      if ( tsosNew.isValid() ) {
    //        cout << "New state outside  = " << tsosNew.localPosition() 
    //             << " " << tsosNew.localMomentum() << endl;
    //        if ( tsosNew.hasError() )
    //  	cout << "New errors at outside = " << tsosNew.localError().matrix() << endl;
    //      }
    //
    // Second part: backward propagation towards the origin
    //
    cout << "Changing direction" << endl;
    oldPropagator->setPropagationDirection(oppositeToMomentum);
    if ( !theAnyDirection )  
      newPropagator->setPropagationDirection(oppositeToMomentum);
    while ( true ) {
      //
      // Make one step, use pathlength to check if starting point was passed.
      //   Get true position / direction and generate plane.
      //
      if ( (stot=(trajectoryGenerator->randomStepBackward(theMaxStep)-stot0))<0. )  break;
      xGen = trajectoryGenerator->position();
      pGen = trajectoryGenerator->momentum();
      //          cout << " stepped to x = " << xGen << endl;
      //          cout << "            p = " << pGen << endl;
      //        RandomPlaneGenerator::PlanePtr currentPlane = (*surfaceGenerator)(xGen,pGen); 
      RandomPlaneGenerator::PlanePtr currentPlane(0);
      BoundCylinder::BoundCylinderPointer currentCylinder(0);
      if ( surfaceIsPlane )
	currentPlane = (*planeGenerator)(xGen,pGen);
      else
	currentCylinder = (*cylinderGenerator)(xGen,pGen);
      //       if ( theMaterialEffects && theSurfaceType!="cylinder" ) {
      // 	currentPlane->setMediumProperties(new MediumProperties(theRadLength,theXi));
      //       }
      // continue propagation of states by propagator 1
      if ( tsosOld.isValid() ) {
	if ( surfaceIsPlane )
	  tsosOld = oldPropagator->propagate(tsosOld,*currentPlane);
	else
	  tsosOld = oldPropagator->propagate(tsosOld,*currentCylinder);
	if ( !tsosOld.isValid() )  cout << "  tsosOld is invalid" << endl;
      }
      // continue propagation of states by propagator 2
      if ( tsosNew.isValid() ) {
	if ( surfaceIsPlane )
	  tsosNew = newPropagator->propagate(tsosNew,*currentPlane);
	else
	  tsosNew = newPropagator->propagate(tsosNew,*currentCylinder);
	if ( !tsosNew.isValid() )  cout << "  tsosNew is invalid" << endl;
	// 	else  cout << tsosNew.localParameters().position() << " "
	// 		   << tsosNew.localParameters().momentum() << " "
	// 		   << tsosNew.localParameters().charge() << " "
	// 		   << tsosNew.localParameters().pzSign() << endl;
      }
      // add results to ntuple
      if ( surfaceIsPlane )
	theHistogrammer.addStep(stot,xGen,pGen,
				currentPlane->position(),
				currentPlane->toGlobal(LocalVector(0.,0.,1.)),
				tsosOld,tsosNew);
      else {
	GlobalPoint pos = currentCylinder->position();
	float radius = currentCylinder->radius();
	theHistogrammer.addStep(stot,xGen,pGen,
				GlobalPoint(pos.x(),pos.y(),radius),
				GlobalVector(xGen.x(),xGen.y(),0.).unit(),
				tsosOld,tsosNew);
      }
    } 
    //
    // Back close to starting point: final step towards plane at starting point
    //
    //      cout << "PathLength at end = " << stot << endl;
    // continue propagation of states by propagator 1
    if ( tsosOld.isValid() ) {
      if ( surfaceIsPlane )
	tsosOld = oldPropagator->propagate(tsosOld,*startPlane);
      else
	tsosOld = oldPropagator->propagate(tsosOld,*startCylinder);
      //  	if ( !tsosOld.isValid() )
      //  	  cout << "  tsosOld is invalid" << endl;
      //  	else {
      //  	  cout << "Old state at start = " << tsosOld.localPosition() 
      //             << " " << tsosOld.localMomentum() << endl;
      //  	  if ( tsosOld.hasError() )
      //  	    cout << "Old errors back at start = " 
      //               << tsosOld.localError().matrix() << endl;
      //  	}
    }
    // continue propagation of states by propagator 2
    if ( tsosNew.isValid() ) {
      if ( surfaceIsPlane )
	tsosNew = newPropagator->propagate(tsosNew,*startPlane);
      else
	tsosNew = newPropagator->propagate(tsosNew,*startCylinder);
      //       cout << tsosNew.localParameters().position() << " "
      // 	   << tsosNew.localParameters().momentum() << " "
      // 	   << tsosNew.localParameters().charge() << " "
      // 	   << tsosNew.localParameters().pzSign() << endl;
      //  	if ( !tsosNew.isValid() )
      //  	  cout << "  tsosNew is invalid" << endl;
      //  	else {
      //  	  cout << "New state at start = " << tsosNew.localPosition() 
      //             << " " << tsosNew.localMomentum() << endl;
      //  	  if ( tsosNew.hasError() )
      //  	    cout << "New errors back at start = " 
      //               << tsosNew.localError().matrix() << endl;
      //  	}
    }
    // add results to ntuple
    if ( surfaceIsPlane ) 
      theHistogrammer.addStep(stot,xStart,pStart,
			      startPlane->position(),
			      startPlane->toGlobal(LocalVector(0.,0.,1.)),
			      tsosOld,tsosNew);
    else {
      GlobalPoint pos = startCylinder->position();
      float radius = startCylinder->radius();
      theHistogrammer.addStep(stot,xStart,pStart,
			      GlobalPoint(pos.x(),pos.y(),radius),
			      GlobalVector(xStart.x(),xStart.y(),0.).unit(),
			      tsosOld,tsosNew);
    }
    cout << endl;
    // terminate ntuple entry
    theHistogrammer.fill();
    cout << "After event " << ievt << endl;
  }

  if ( planeGenerator )  delete planeGenerator;
  if ( cylinderGenerator )  delete cylinderGenerator;
  delete oldPropagator;
  delete newPropagator;
}
