#include "Validation/Geometry/interface/MaterialBudgetTree.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


MaterialBudgetTree::MaterialBudgetTree(std::shared_ptr<MaterialBudgetData> data, const std::string& filename )
  : MaterialBudgetFormat( data )
{
  theFile = std::make_unique<TFile>(filename.c_str(),"RECREATE");
  theFile->cd();
  book();
}


void MaterialBudgetTree::book() 
{
  LogDebug("MaterialBudget") << "MaterialBudgetTree: Booking user TTree";
  // create the TTree
  theTree = std::make_unique<TTree>("T1","GeometryTest Tree");

  // GENERAL block
  theTree->Branch("MB", &t_MB, "MB/F");
  theTree->Branch("IL", &t_IL, "IL/F");
  
  // PARTICLE Block
  theTree->Branch( "Particle ID",     &t_ParticleID,     "Particle_ID/I"  );
  theTree->Branch( "Particle Pt",     &t_ParticlePt,     "Particle_Pt/F"  );
  theTree->Branch( "Particle Eta",    &t_ParticleEta,    "Particle_Eta/F" );
  theTree->Branch( "Particle Phi",    &t_ParticlePhi,    "Particle_Phi/F" );
  theTree->Branch( "Particle Energy", &t_ParticleEnergy, "Particle_E/F"   );
  theTree->Branch( "Particle Mass",   &t_ParticleMass,   "Particle_M/F"   );
  
  if( theData->allStepsON() ) {
    theTree->Branch("Nsteps", &t_Nsteps, "Nsteps/I");
    theTree->Branch("DeltaMB", t_DeltaMB, "DeltaMB[Nsteps]/F");
    theTree->Branch("DeltaMB_SUP", t_DeltaMB_SUP, "DeltaMB_SUP[Nsteps]/F");
    theTree->Branch("DeltaMB_SEN", t_DeltaMB_SEN, "DeltaMB_SEN[Nsteps]/F");
    theTree->Branch("DeltaMB_CAB", t_DeltaMB_CAB, "DeltaMB_CAB[Nsteps]/F");
    theTree->Branch("DeltaMB_COL", t_DeltaMB_COL, "DeltaMB_COL[Nsteps]/F");
    theTree->Branch("DeltaMB_ELE", t_DeltaMB_ELE, "DeltaMB_ELE[Nsteps]/F");
    theTree->Branch("DeltaMB_OTH", t_DeltaMB_OTH, "DeltaMB_OTH[Nsteps]/F");
    theTree->Branch("DeltaMB_AIR", t_DeltaMB_AIR, "DeltaMB_AIR[Nsteps]/F");

    theTree->Branch("DeltaIL", t_DeltaIL, "DeltaIL[Nsteps]/F");
    theTree->Branch("DeltaIL_SUP", t_DeltaIL_SUP, "DeltaIL_SUP[Nsteps]/F");
    theTree->Branch("DeltaIL_SEN", t_DeltaIL_SEN, "DeltaIL_SEN[Nsteps]/F");
    theTree->Branch("DeltaIL_CAB", t_DeltaIL_CAB, "DeltaIL_CAB[Nsteps]/F");
    theTree->Branch("DeltaIL_COL", t_DeltaIL_COL, "DeltaIL_COL[Nsteps]/F");
    theTree->Branch("DeltaIL_ELE", t_DeltaIL_ELE, "DeltaIL_ELE[Nsteps]/F");
    theTree->Branch("DeltaIL_OTH", t_DeltaIL_OTH, "DeltaIL_OTH[Nsteps]/F");
    theTree->Branch("DeltaIL_AIR", t_DeltaIL_AIR, "DeltaIL_AIR[Nsteps]/F");

    theTree->Branch("Initial X", t_InitialX, "Initial_X[Nsteps]/D");
    theTree->Branch("Initial Y", t_InitialY, "Initial_Y[Nsteps]/D");
    theTree->Branch("Initial Z", t_InitialZ, "Initial_Z[Nsteps]/D");

    theTree->Branch("Final X",   t_FinalX,   "Final_X[Nsteps]/D");
    theTree->Branch("Final Y",   t_FinalY,   "Final_Y[Nsteps]/D");
    theTree->Branch("Final Z",   t_FinalZ,   "Final_Z[Nsteps]/D");

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
    
    theTree->Branch("Material ID",      t_MaterialID,      "MaterialID[Nsteps]/I");
    theTree->Branch("Material Name",    t_MaterialName,    "MaterialName[Nsteps]/C");
    theTree->Branch("Material X0",      t_MaterialX0,      "MaterialX0[Nsteps]/F");
    theTree->Branch("Material Lambda0", t_MaterialLambda0, "MaterialLambda0[Nsteps]/F");
    theTree->Branch("Material Density", t_MaterialDensity, "MaterialDensity[Nsteps]/F");
    
    theTree->Branch("Particle Step ID",               t_ParticleStepID,              "Step_ID[Nsteps]/I");
    theTree->Branch("Particle Step Initial Pt",       t_ParticleStepInitialPt,       "Step_Initial_Pt[Nsteps]/F");
    theTree->Branch("Particle Step Initial Eta",      t_ParticleStepInitialEta,      "Step_Initial_Eta[Nsteps]/F");
    theTree->Branch("Particle Step Initial Phi",      t_ParticleStepInitialPhi,      "Step_Initial_Phi[Nsteps]/F");
    theTree->Branch("Particle Step Initial Energy",   t_ParticleStepInitialEnergy,   "Step_Initial_E[Nsteps]/F");
    theTree->Branch("Particle Step Initial Px",       t_ParticleStepInitialPx,       "Step_Initial_Px[Nsteps]/F");
    theTree->Branch("Particle Step Initial Py",       t_ParticleStepInitialPy,       "Step_Initial_Py[Nsteps]/F");
    theTree->Branch("Particle Step Initial Pz",       t_ParticleStepInitialPz,       "Step_Initial_Pz[Nsteps]/F");
    theTree->Branch("Particle Step Initial Beta",     t_ParticleStepInitialBeta,     "Step_Initial_Beta[Nsteps]/F");
    theTree->Branch("Particle Step Initial Gamma",    t_ParticleStepInitialGamma,    "Step_Initial_Gamma[Nsteps]/F");
    theTree->Branch("Particle Step Initial Mass",     t_ParticleStepInitialMass,     "Step_Initial_Mass[Nsteps]/F");
    theTree->Branch("Particle Step Final Pt",         t_ParticleStepFinalPt,         "Step_Final_Pt[Nsteps]/F");
    theTree->Branch("Particle Step Final Eta",        t_ParticleStepFinalEta,        "Step_Final_Eta[Nsteps]/F");
    theTree->Branch("Particle Step Final Phi",        t_ParticleStepFinalPhi,        "Step_Final_Phi[Nsteps]/F");
    theTree->Branch("Particle Step Final Energy",     t_ParticleStepFinalEnergy,     "Step_Final_E[Nsteps]/F");
    theTree->Branch("Particle Step Final Px",         t_ParticleStepFinalPx,         "Step_Final_Px[Nsteps]/F");
    theTree->Branch("Particle Step Final Py",         t_ParticleStepFinalPy,         "Step_Final_Py[Nsteps]/F");
    theTree->Branch("Particle Step Final Pz",         t_ParticleStepFinalPz,         "Step_Final_Pz[Nsteps]/F");
    theTree->Branch("Particle Step Final Beta",       t_ParticleStepFinalBeta,       "Step_Final_Beta[Nsteps]/F");
    theTree->Branch("Particle Step Final Gamma",      t_ParticleStepFinalGamma,      "Step_Final_Gamma[Nsteps]/F");
    theTree->Branch("Particle Step Final Mass",       t_ParticleStepFinalMass,       "Step_Final_Mass[Nsteps]/F");
    theTree->Branch("Particle Step Pre Interaction",  t_ParticleStepPreInteraction,  "Step_PreInteraction[Nsteps]/I");
    theTree->Branch("Particle Step Post Interaction", t_ParticleStepPostInteraction, "Step_PostInteraction[Nsteps]/I");

  }
  
  LogDebug("MaterialBudget") << "MaterialBudgetTree: Booking user TTree done";
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
  t_IL  = theData->getTotalIL();
  //  t_Eta = theData->getEta();
  //  t_Phi = theData->getPhi();

  t_ParticleID     = theData->getID();
  t_ParticlePt     = theData->getPt();
  t_ParticleEta    = theData->getEta();
  t_ParticlePhi    = theData->getPhi();
  t_ParticleEnergy = theData->getEnergy();
  t_ParticleMass   = theData->getMass();
  
  if( theData->allStepsON() ) {

    t_Nsteps = theData->getNumberOfSteps();
    
    if( t_Nsteps > MAXSTEPS ) t_Nsteps = MAXSTEPS;

    edm::LogInfo("MaterialBudget") << "MaterialBudgetTree: Number of Steps into the tree " << t_Nsteps;

    for(int ii=0;ii<t_Nsteps;ii++) {

      t_DeltaMB[ii] = theData->getStepDmb(ii);
      t_DeltaMB_SUP[ii] = theData->getSupportDmb(ii);
      t_DeltaMB_SEN[ii] = theData->getSensitiveDmb(ii);
      t_DeltaMB_CAB[ii] = theData->getCablesDmb(ii);
      t_DeltaMB_COL[ii] = theData->getCoolingDmb(ii);
      t_DeltaMB_ELE[ii] = theData->getElectronicsDmb(ii);
      t_DeltaMB_OTH[ii] = theData->getOtherDmb(ii);
      t_DeltaMB_AIR[ii] = theData->getAirDmb(ii);
      
      t_DeltaIL[ii] = theData->getStepDil(ii);
      t_DeltaIL_SUP[ii] = theData->getSupportDil(ii);
      t_DeltaIL_SEN[ii] = theData->getSensitiveDil(ii);
      t_DeltaIL_CAB[ii] = theData->getCablesDil(ii);
      t_DeltaIL_COL[ii] = theData->getCoolingDil(ii);
      t_DeltaIL_ELE[ii] = theData->getElectronicsDil(ii);
      t_DeltaIL_OTH[ii] = theData->getOtherDil(ii);
      t_DeltaIL_AIR[ii] = theData->getAirDil(ii);

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
      
      t_MaterialID[ii]        = theData->getStepMaterialID(ii);
      t_MaterialName[ii]      = theData->getStepMaterialName(ii).c_str();
      t_MaterialX0[ii]        = theData->getStepMaterialX0(ii);
      t_MaterialLambda0[ii]   = theData->getStepMaterialLambda0(ii);
      t_MaterialDensity[ii]   = theData->getStepMaterialDensity(ii);
      
      t_ParticleStepID[ii]              = theData->getStepID(ii);
      t_ParticleStepInitialPt[ii]       = theData->getStepInitialPt(ii);
      t_ParticleStepInitialEta[ii]      = theData->getStepInitialEta(ii);
      t_ParticleStepInitialPhi[ii]      = theData->getStepInitialPhi(ii);
      t_ParticleStepInitialEnergy[ii]   = theData->getStepInitialEnergy(ii);
      t_ParticleStepInitialPx[ii]       = theData->getStepInitialPx(ii);
      t_ParticleStepInitialPy[ii]       = theData->getStepInitialPy(ii);
      t_ParticleStepInitialPz[ii]       = theData->getStepInitialPz(ii);
      t_ParticleStepInitialBeta[ii]     = theData->getStepInitialBeta(ii);
      t_ParticleStepInitialGamma[ii]    = theData->getStepInitialGamma(ii);
      t_ParticleStepInitialMass[ii]     = theData->getStepInitialMass(ii);
      t_ParticleStepFinalPt[ii]         = theData->getStepFinalPt(ii);
      t_ParticleStepFinalEta[ii]        = theData->getStepFinalEta(ii);
      t_ParticleStepFinalPhi[ii]        = theData->getStepFinalPhi(ii);
      t_ParticleStepFinalEnergy[ii]     = theData->getStepFinalEnergy(ii);
      t_ParticleStepFinalPx[ii]         = theData->getStepFinalPx(ii);
      t_ParticleStepFinalPy[ii]         = theData->getStepFinalPy(ii);
      t_ParticleStepFinalPz[ii]         = theData->getStepFinalPz(ii);
      t_ParticleStepFinalBeta[ii]       = theData->getStepFinalBeta(ii);
      t_ParticleStepFinalGamma[ii]      = theData->getStepFinalGamma(ii);
      t_ParticleStepFinalMass[ii]       = theData->getStepFinalMass(ii);
      t_ParticleStepPreInteraction[ii]  = theData->getStepPreProcess(ii);
      t_ParticleStepPostInteraction[ii] = theData->getStepPostProcess(ii);
      
    }
  }

  theTree->Fill();
}


void MaterialBudgetTree::endOfRun() 
{

  // Prefered method to include any instruction
  // once all the tracks are done

  edm::LogInfo("MaterialBudget") << "MaterialBudgetTree Writing TTree to ROOT file";

  theFile->cd();
  theTree->Write();
  theFile->Close();
}
