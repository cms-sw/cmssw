#include "TrackPropagation/NavGeometry/test/stubs/PropagatorTestTree.h"

#include "TFile.h"
#include "TTree.h"

PropagatorTestTree::PropagatorTestTree (const char* file) : 
theMaxSteps(1000)
{
  //
  // Open output file
  //
  theFile = new TFile(file,"RECREATE");
  theFile->SetCompressionLevel(2);
  //
  // Create tree
  //
  createArrays();
  bookTree();
}

PropagatorTestTree::~PropagatorTestTree () 
{
  //
  // termination
  //
  theTree->GetDirectory()->cd();
  theTree->Write();
  delete theTree;
  delete theFile;
}

void PropagatorTestTree::createArrays()
{
  thePathLengths = new float[theMaxSteps];

  createFloatArrays(theGeneratedPoints);
  createFloatArrays(theGeneratedMomenta);

  createFloatArrays(thePlanePoints);
  createFloatArrays(thePlaneNormals);

  theOldStatus = new unsigned char[theMaxSteps];
  createFloatArrays(theOldDPoints);
  createFloatArrays(theOldDMomenta);

  theNewStatus = new unsigned char[theMaxSteps];
  createFloatArrays(theNewDPoints);
  createFloatArrays(theNewDMomenta);
}

void PropagatorTestTree::removeArrays()
{
  delete [] thePathLengths;

  removeFloatArrays(theGeneratedPoints);
  removeFloatArrays(theGeneratedMomenta);
  removeFloatArrays(thePlanePoints);
  removeFloatArrays(thePlaneNormals);

  delete [] theOldStatus;
  removeFloatArrays(theOldDPoints);
  removeFloatArrays(theOldDMomenta);

  delete [] theNewStatus;
  removeFloatArrays(theNewDPoints);
  removeFloatArrays(theNewDMomenta);
}

void 
PropagatorTestTree::createFloatArrays (float* arrayPtr[3]) const
{
  for ( int i=0; i<3; i++ )
    arrayPtr[i] = new float[theMaxSteps];
}

void 
PropagatorTestTree::removeFloatArrays (float* arrayPtr[3]) const
{
  for ( int i=0; i<3; i++ )  delete [] arrayPtr[i];
}

void PropagatorTestTree::bookTree() {
  //
  // Tree
  //
  theTree = new TTree("propagation","Propagator Test");
  //
  // Branches
  //
  // Helix parameters:
  //   Start position and direction, helix center,
  //   curvature and charge
  //
  theTree->Branch("HelixStart",theStart,"X/F:Y/F:Z/F:TH/F:PH/F:PT/F");
  theTree->Branch("HelixCenter",theCenter,"X/F:Y/F:Z/F");
  theTree->Branch("HelixPars",&theRho,"Rho/F:Q/B");
  //
  // Values / step:
  //   total path length from start, generated position and direction,
  //   plane position and normal vector, and for both propagators:
  //     status, position and direction
  //
  theTree->Branch("NStep_",&theNrOfSteps,"NStep/i:NStepFwd/i");
  theTree->Branch("Path",thePathLengths,"Stotal[NStep]/F");

  theTree->Branch("XGen_",theGeneratedPoints[0],"XGen[NStep]/F");
  theTree->Branch("YGen_",theGeneratedPoints[1],"YGen[NStep]/F");
  theTree->Branch("ZGen_",theGeneratedPoints[2],"ZGen[NStep]/F");
  theTree->Branch("THGen_",theGeneratedMomenta[0],"THGen[NStep]/F");
  theTree->Branch("PHGen_",theGeneratedMomenta[1],"PHGen[NStep]/F");
  theTree->Branch("PTGen_",theGeneratedMomenta[2],"PTGen[NStep]/F");

  theTree->Branch("XPlane_",thePlanePoints[0],"XPlane[NStep]/F");
  theTree->Branch("YPlane_",thePlanePoints[1],"YPlane[NStep]/F");
  theTree->Branch("ZPlane_",thePlanePoints[2],"ZPlane[NStep]/F");
  theTree->Branch("XNPlane_",thePlaneNormals[0],"XNPlane[NStep]/F");
  theTree->Branch("YNPlane_",thePlaneNormals[1],"YNPlane[NStep]/F");
  theTree->Branch("ZNPlane_",thePlaneNormals[2],"ZNPlane[NStep]/F");

  theTree->Branch("OldStat_",theOldStatus,"OldStat[NStep]/b");
  theTree->Branch("DXOld_",theOldDPoints[0],"DXOld[NStep]/F");
  theTree->Branch("DYOld_",theOldDPoints[1],"DYOld[NStep]/F");
  theTree->Branch("DZOld_",theOldDPoints[2],"DZOld[NStep]/F");
  theTree->Branch("DTHOld_",theOldDMomenta[0],"DTHOld[NStep]/F");
  theTree->Branch("DPHOld_",theOldDMomenta[1],"DPHOld[NStep]/F");
  theTree->Branch("DPTOld_",theOldDMomenta[2],"DPTOld[NStep]/F");

  theTree->Branch("NewStat_",theNewStatus,"NewStat[NStep]/b");
  theTree->Branch("DXNew_",theNewDPoints[0],"DXNew[NStep]/F");
  theTree->Branch("DYNew_",theNewDPoints[1],"DYNew[NStep]/F");
  theTree->Branch("DZNew_",theNewDPoints[2],"DZNew[NStep]/F");
  theTree->Branch("DTHNew_",theNewDMomenta[0],"DTHNew[NStep]/F");
  theTree->Branch("DPHNew_",theNewDMomenta[1],"DPHNew[NStep]/F");
  theTree->Branch("DPTNew_",theNewDMomenta[2],"DPTNew[NStep]/F");
}

void PropagatorTestTree::fillHelix (const GlobalPoint& point,
				    const GlobalVector& momentum,
				    const GlobalPoint& center,
				    const float curvature,
				    const int charge) {
  //
  // store helix parameters
  //  
  theStart[0] = point.x();
  theStart[1] = point.y();
  theStart[2] = point.z();
  theStart[3] = momentum.theta();
  theStart[4] = momentum.phi();
  theStart[5] = momentum.perp();
  theCenter[0] = center.x();
  theCenter[1] = center.y();
  theCenter[2] = center.z();
  theRho = curvature;
  theCharge = (char)charge;
  //
  // reset internal arrays
  //
  theNrOfSteps = 0;
  theNrOfFwdSteps = 0;
  thePreviousPathLength = -1.;
}

void PropagatorTestTree::addStep (const float pathLength,
				  const GlobalPoint& xGen, const GlobalVector& pGen,
				  const GlobalPoint& xPlane, const GlobalVector& nPlane,
				  const TrajectoryStateOnSurface& oldState,
				  const TrajectoryStateOnSurface& newState) 
{
  //
  // check #points
  //
  if ( theNrOfSteps>=theMaxSteps )  return;
  //
  // information about generated point
  //
  thePathLengths[theNrOfSteps] = pathLength;
  storeCartesian(xGen,theGeneratedPoints,theNrOfSteps);
  storeAngles(pGen,theGeneratedMomenta,theNrOfSteps);
  //
  // information about plane
  //
  storeCartesian(xPlane,thePlanePoints,theNrOfSteps);
  storeCartesian(nPlane,thePlaneNormals,theNrOfSteps);
  //
  // first propagator
  //
  theOldStatus[theNrOfSteps] = oldState.isValid() ? 1 : 0;
  if ( oldState.isValid() ) {
    storeCartesian(oldState.globalPosition(),xGen,theOldDPoints,theNrOfSteps);
    storeAngles(oldState.globalMomentum(),pGen,theOldDMomenta,theNrOfSteps);
  }
  else {
    theOldDPoints[0][theNrOfSteps] = 999.;
    theOldDPoints[1][theNrOfSteps] = 999.;
    theOldDPoints[2][theNrOfSteps] = 999.;
    theOldDMomenta[0][theNrOfSteps] = 999.;
    theOldDMomenta[1][theNrOfSteps] = 999.;
    theOldDMomenta[2][theNrOfSteps] = 999.*pGen.perp();
  }
  //
  // second propagator
  //
  theNewStatus[theNrOfSteps] = newState.isValid() ? 1 : 0;
  if ( newState.isValid() ) {
    storeCartesian(newState.globalPosition(),xGen,theNewDPoints,theNrOfSteps);
    storeAngles(newState.globalMomentum(),pGen,theNewDMomenta,theNrOfSteps);
  }
  else {
    theNewDPoints[0][theNrOfSteps] = 999.;
    theNewDPoints[1][theNrOfSteps] = 999.;
    theNewDPoints[2][theNrOfSteps] = 999.;
    theNewDMomenta[0][theNrOfSteps] = 999.;
    theNewDMomenta[1][theNrOfSteps] = 999.;
    theNewDMomenta[2][theNrOfSteps] = 999.*pGen.perp();
  }
  //
  // increase step count
  //
  theNrOfSteps++;
  //
  // forward / backward
  //
  if ( pathLength>thePreviousPathLength )  theNrOfFwdSteps++;
  thePreviousPathLength = pathLength;
}

void
PropagatorTestTree::fill ()
{
  theTree->Fill();
}
