#include "Validation/Geometry/interface/MaterialBudgetTree.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"


MaterialBudgetTree::MaterialBudgetTree(MaterialBudgetData* data, const std::string& filename ): MaterialBudgetFormat( data )
{

  theFile = new TFile(filename.c_str(),"RECREATE");
  
  theFile->cd();

  book();


}


void MaterialBudgetTree::book() 
{
  std::cout << "=== booking user TTree ===" << std::endl;
  // create the TTree
  theTree = new TTree("T1","GeometryTest Tree");

  // GENERAL block
  theTree->Branch("MB", &t_MB, "MB/F");
  theTree->Branch("Eta", &t_Eta, "Eta/F");
  theTree->Branch("Phi", &t_Phi, "Phi/F");

 
  if( theData->allStepsON() ) {
    theTree->Branch("Nsteps", &t_Nsteps, "Nsteps/I");
    theTree->Branch("DeltaMB", t_DeltaMB, "DeltaMB[Nsteps]/F");
    theTree->Branch("X", t_X, "X[Nsteps]/F");
    theTree->Branch("Y", t_Y, "Y[Nsteps]/F");
    theTree->Branch("Z", t_Z, "Z[Nsteps]/F");
    theTree->Branch("VoluID", t_VoluID, "VoluID[Nsteps]/I");
    theTree->Branch("MateID", t_MateID, "MateID[Nsteps]/I");
  }

  std::cout << "=== booking user TTree done ===" << std::endl;

}


void MaterialBudgetTree::fillStartTrack()
{

}


void MaterialBudgetTree::fillPerStep()
{
}


void MaterialBudgetTree::fillEndTrack()
{
  t_MB = theData->getTotalMB();
  t_Eta = theData->getEta();
  t_Phi = theData->getPhi();
  
  // do this only if I really want to save all the steps
  if( theData->allStepsON() ) {
    int t_Nsteps = theData->getNumberOfSteps();
    if( t_Nsteps > MAXSTEPS ) t_Nsteps = MAXSTEPS;
    for(int ii=0;ii<t_Nsteps;ii++) {
      t_DeltaMB[ii] = theData->getStepDmb(ii);
      t_X[ii] = theData->getStepX(ii);
      t_Y[ii] = theData->getStepY(ii);
      t_Z[ii] = theData->getStepZ(ii);
      t_VoluID[ii] = theData->getStepVoluId(ii);
      t_MateID[ii] = theData->getStepMateId(ii);
    }
  }

  theTree->Fill();

}


// here one can print the histograms or 
// manipulate them before they are written to file
void MaterialBudgetTree::hend() 
{
  std::cout << " === save user TTree ===" << std::endl;
 
  theFile->cd();
  theTree->Write();
  
  theFile->Close();

}

