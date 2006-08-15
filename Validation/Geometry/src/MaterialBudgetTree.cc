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
    // rr
    theTree->Branch("DeltaMB_SUP", t_DeltaMB_SUP, "DeltaMB_SUP[Nsteps]/F");
    theTree->Branch("DeltaMB_SEN", t_DeltaMB_SEN, "DeltaMB_SEN[Nsteps]/F");
    theTree->Branch("DeltaMB_CAB", t_DeltaMB_CAB, "DeltaMB_CAB[Nsteps]/F");
    theTree->Branch("DeltaMB_COL", t_DeltaMB_COL, "DeltaMB_COL[Nsteps]/F");
    theTree->Branch("DeltaMB_ELE", t_DeltaMB_ELE, "DeltaMB_ELE[Nsteps]/F");
    theTree->Branch("DeltaMB_OTH", t_DeltaMB_OTH, "DeltaMB_OTH[Nsteps]/F");
    theTree->Branch("DeltaMB_AIR", t_DeltaMB_AIR, "DeltaMB_AIR[Nsteps]/F");
    // rr
    theTree->Branch("Initial X", t_InitialX, "Initial_X[Nsteps]/D");
    theTree->Branch("Initial Y", t_InitialY, "Initial_Y[Nsteps]/D");
    theTree->Branch("Initial Z", t_InitialZ, "Initial_Z[Nsteps]/D");
    theTree->Branch("Final X",   t_FinalX,   "Final_X[Nsteps]/D");
    theTree->Branch("Final Y",   t_FinalY,   "Final_Y[Nsteps]/D");
    theTree->Branch("Final Z",   t_FinalZ,   "Final_Z[Nsteps]/D");
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
    
    theTree->Branch("Particle Step ID",             t_ParticleStepID,            "Step_ID[Nsteps]/I");
    theTree->Branch("Particle Step Initial Pt",     t_ParticleStepInitialPt,     "Step_Initial_Pt[Nsteps]/F");
    theTree->Branch("Particle Step Initial Eta",    t_ParticleStepInitialEta,    "Step_Initial_Eta[Nsteps]/F");
    theTree->Branch("Particle Step Initial Phi",    t_ParticleStepInitialPhi,    "Step_Initial_Phi[Nsteps]/F");
    theTree->Branch("Particle Step Initial Energy", t_ParticleStepInitialEnergy, "Step_Initial_E[Nsteps]/F");
    theTree->Branch("Particle Step Final Pt",       t_ParticleStepFinalPt,       "Step_Final_Pt[Nsteps]/F");
    theTree->Branch("Particle Step Final Eta",      t_ParticleStepFinalEta,      "Step_Final_Eta[Nsteps]/F");
    theTree->Branch("Particle Step Final Phi",      t_ParticleStepFinalPhi,      "Step_Final_Phi[Nsteps]/F");
    theTree->Branch("Particle Step Final Energy",   t_ParticleStepFinalEnergy,   "Step_Final_E[Nsteps]/F");
    theTree->Branch("Particle Step Interaction",    t_ParticleStepInteraction,   "Step_Interaction[Nsteps]/I");
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
      t_DeltaMB_SUP[ii] = theData->getSupportDmb(ii);
      t_DeltaMB_SEN[ii] = theData->getSensitiveDmb(ii);
      t_DeltaMB_CAB[ii] = theData->getCablesDmb(ii);
      t_DeltaMB_COL[ii] = theData->getCoolingDmb(ii);
      t_DeltaMB_ELE[ii] = theData->getElectronicsDmb(ii);
      t_DeltaMB_OTH[ii] = theData->getOtherDmb(ii);
      t_DeltaMB_AIR[ii] = theData->getAirDmb(ii);
      
      t_InitialX[ii] = theData->getStepInitialX(ii);
      t_InitialY[ii] = theData->getStepInitialY(ii);
      t_InitialZ[ii] = theData->getStepInitialZ(ii);
      t_FinalX[ii]   = theData->getStepFinalX(ii);
      t_FinalY[ii]   = theData->getStepFinalY(ii);
      t_FinalZ[ii]   = theData->getStepFinalZ(ii);
      
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
      
      t_ParticleStepID[ii]            = theData->getStepID(ii);
      t_ParticleStepInitialPt[ii]     = theData->getStepInitialPt(ii);
      t_ParticleStepInitialEta[ii]    = theData->getStepInitialEta(ii);
      t_ParticleStepInitialPhi[ii]    = theData->getStepInitialPhi(ii);
      t_ParticleStepInitialEnergy[ii] = theData->getStepInitialEnergy(ii);
      t_ParticleStepFinalPt[ii]       = theData->getStepFinalPt(ii);
      t_ParticleStepFinalEta[ii]      = theData->getStepFinalEta(ii);
      t_ParticleStepFinalPhi[ii]      = theData->getStepFinalPhi(ii);
      t_ParticleStepFinalEnergy[ii]   = theData->getStepFinalEnergy(ii);
      t_ParticleStepInteraction[ii]   = theData->getStepProcess(ii);
      
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

