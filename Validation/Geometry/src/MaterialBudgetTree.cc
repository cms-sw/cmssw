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
  
  // rr
  // PARTICLE Block
  theTree->Branch( "Particle ID",     &t_ParticleID,     "Particle_ID/I"  );
  theTree->Branch( "Particle Pt",     &t_ParticlePt,     "Particle_Pt/F"  );
  theTree->Branch( "Particle Eta",    &t_ParticleEta,    "Particle_Eta/F" );
  theTree->Branch( "Particle Phi",    &t_ParticlePhi,    "Particle_Phi/F" );
  theTree->Branch( "Particle Energy", &t_ParticleEnergy, "Particle_E/F"   );
  // rr
 
  if( theData->allStepsON() ) {
    theTree->Branch("Nsteps", &t_Nsteps, "Nsteps/I");
    theTree->Branch("DeltaMB", t_DeltaMB, "DeltaMB[Nsteps]/F");
    theTree->Branch("X", t_X, "X[Nsteps]/F");
    theTree->Branch("Y", t_Y, "Y[Nsteps]/F");
    theTree->Branch("Z", t_Z, "Z[Nsteps]/F");
    //    theTree->Branch("VoluID", t_VoluID, "VoluID[Nsteps]/I");
    //    theTree->Branch("MateID", t_MateID, "MateID[Nsteps]/I");
    // rr
    theTree->Branch("Volume ID",       t_VolumeID,     "VolumeID[Nsteps]/I");
    theTree->Branch("Volume Name",     t_VolumeName,   "VolumeName[Nsteps]/C");
    theTree->Branch("Volume Copy",     t_VolumeCopy,   "VolumeCopy[Nsteps]/I");
    theTree->Branch("Volume X",        t_VolumeX,      "VolumeX[Nsteps]/F");
    theTree->Branch("Volume Y",        t_VolumeY,      "VolumeY[Nsteps]/F");
    theTree->Branch("Volume Z",        t_VolumeZ,      "VolumeZ[Nsteps]/F");
    theTree->Branch("Volume X axis 1", t_VolumeXaxis1, "VolumeXaxis1[Nsteps]/F");
    theTree->Branch("Volume X axis 2", t_VolumeXaxis2, "VolumeXaxis2[Nsteps]/F");
    theTree->Branch("Volume X axis 3", t_VolumeXaxis3, "VolumeXaxis3[Nsteps]/F");
    theTree->Branch("Volume Y axis 1", t_VolumeYaxis1, "VolumeYaxis1[Nsteps]/F");
    theTree->Branch("Volume Y axis 2", t_VolumeYaxis2, "VolumeYaxis2[Nsteps]/F");
    theTree->Branch("Volume Y axis 3", t_VolumeYaxis3, "VolumeYaxis3[Nsteps]/F");
    theTree->Branch("Volume Z axis 1", t_VolumeZaxis1, "VolumeZaxis1[Nsteps]/F");
    theTree->Branch("Volume Z axis 2", t_VolumeZaxis2, "VolumeZaxis2[Nsteps]/F");
    theTree->Branch("Volume Z axis 3", t_VolumeZaxis3, "VolumeZaxis3[Nsteps]/F");

    theTree->Branch("Material ID",   t_MaterialID,   "MaterialID[Nsteps]/I");
    theTree->Branch("Material Name", t_MaterialName, "MaterialName[Nsteps]/C");
    theTree->Branch("Material X0",   t_MaterialX0,   "MaterialX0[Nsteps]/F");

    theTree->Branch("Particle Step ID",     t_ParticleStepID,     "Step_ID[Nsteps]/I");
    theTree->Branch("Particle Step Pt",     t_ParticleStepPt,     "Step_Pt[Nsteps]/F");
    theTree->Branch("Particle Step Eta",    t_ParticleStepEta,    "Step_Eta[Nsteps]/F");
    theTree->Branch("Particle Step Phi",    t_ParticleStepPhi,    "Step_Phi[Nsteps]/F");
    theTree->Branch("Particle Step Energy", t_ParticleStepEnergy, "Step_E[Nsteps]/F");
    // rr
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
  t_MB  = theData->getTotalMB();
  //  t_Eta = theData->getEta();
  //  t_Phi = theData->getPhi();

  // rr
  t_ParticleID     = theData->getID();
  t_ParticlePt     = theData->getPt();
  t_ParticleEta    = theData->getEta();
  t_ParticlePhi    = theData->getPhi();
  t_ParticleEnergy = theData->getEnergy();
  // rr
  
  // do this only if I really want to save all the steps
  if( theData->allStepsON() ) {
    t_Nsteps = theData->getNumberOfSteps();
    if( t_Nsteps > MAXSTEPS ) t_Nsteps = MAXSTEPS;
    std::cout << " Number of Steps into the tree " << t_Nsteps << std::endl;
    for(int ii=0;ii<t_Nsteps;ii++) {
      t_DeltaMB[ii] = theData->getStepDmb(ii);
      t_X[ii] = theData->getStepX(ii);
      t_Y[ii] = theData->getStepY(ii);
      t_Z[ii] = theData->getStepZ(ii);
      
      t_VolumeID[ii]     = theData->getStepVolumeID(ii);
      t_VolumeName[ii]   = theData->getStepVolumeName(ii).c_str();
      t_VolumeCopy[ii]   = theData->getStepVolumeCopy(ii);
      t_VolumeX[ii]      = theData->getStepVolumeX(ii);
      t_VolumeY[ii]      = theData->getStepVolumeY(ii);
      t_VolumeZ[ii]      = theData->getStepVolumeZ(ii);
      t_VolumeXaxis1[ii] = theData->getStepVolumeXaxis(ii).x();
      t_VolumeXaxis2[ii] = theData->getStepVolumeXaxis(ii).y();
      t_VolumeXaxis3[ii] = theData->getStepVolumeXaxis(ii).z();
      t_VolumeYaxis1[ii] = theData->getStepVolumeYaxis(ii).x();
      t_VolumeYaxis2[ii] = theData->getStepVolumeYaxis(ii).y();
      t_VolumeYaxis3[ii] = theData->getStepVolumeYaxis(ii).z();
      t_VolumeZaxis1[ii] = theData->getStepVolumeZaxis(ii).x();
      t_VolumeZaxis2[ii] = theData->getStepVolumeZaxis(ii).y();
      t_VolumeZaxis3[ii] = theData->getStepVolumeZaxis(ii).z();

      t_MaterialID[ii]   = theData->getStepMaterialID(ii);
      t_MaterialName[ii] = theData->getStepMaterialName(ii).c_str();
      t_MaterialX0[ii]   = theData->getStepMaterialX0(ii);
      
      t_ParticleStepID[ii]     = theData->getStepID(ii);
      t_ParticleStepPt[ii]     = theData->getStepPt(ii);
      t_ParticleStepEta[ii]    = theData->getStepEta(ii);
      t_ParticleStepPhi[ii]    = theData->getStepPhi(ii);
      t_ParticleStepEnergy[ii] = theData->getStepEnergy(ii);
      
      // rr
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

