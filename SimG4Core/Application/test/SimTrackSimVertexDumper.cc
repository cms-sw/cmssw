// -*- C++ -*-
//
// Package:    SimTrackerDumper
// Class:      SimTrackSimVertexDumper
// 
/*
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//


// system include files
#include <memory>

#include "SimG4Core/Application/test/SimTrackSimVertexDumper.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "TH1F.h"
#include "TH2F.h"

SimTrackSimVertexDumper::SimTrackSimVertexDumper( const edm::ParameterSet& iConfig ):
  HepMCLabel(iConfig.getUntrackedParameter("moduleLabelHepMC",std::string("source"))),
  SimTkLabel(iConfig.getUntrackedParameter("moduleLabelTk",std::string("g4SimHits"))),
  SimVtxLabel(iConfig.getUntrackedParameter("moduleLabelVtx",std::string("g4SimHits"))),
  dumpHepMC(iConfig.getUntrackedParameter("dumpHepMC",bool("false")))
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SimTrackSimVertexDumper::beginJob( const edm::EventSetup& iSetup )
{
  genP                    = fs->make<TH1F>( "genP", "P of generator particles", 100,  0., 300. );
  genPt                   = fs->make<TH1F>( "genPt"       , "Pt of generator particles", 100,  0., 50. );
  genEta                  = fs->make<TH1F>( "genEta"      , "Eta of generator particles", 100,  -8., 8. );
  genPID                  = fs->make<TH1F>( "genPID"      , "Particle ID", 100,  0., 1000. );
  genPIDLossNew           = fs->make<TH1F>( "genPIDLossNew"      , "Particle ID", 100,  0., 1000. );
  genPIDLossOld           = fs->make<TH1F>( "genPIDLossOld"      , "Particle ID", 100,  0., 1000. );
  genZimpact              = fs->make<TH1F>( "genZimpact"  , "Z impact of generator particles on beam pipe", 100,  -800., 800. );
  genPWithOldPtCut        = fs->make<TH1F>( "genPWithOldPtCut", "P of generator particles that satisfy the old Pt cut", 100,  0., 300. );
  genPtWithNewPCut        = fs->make<TH1F>( "genPtWithNewPCut", "Pt of generator particles that satisfy the new P cut", 100,  0., 50. );
  genZWithOldEtaCut       = fs->make<TH1F>( "genZWithOldEtaCut", "Zimpact of generator particles that satisfy the old Eta cut", 100,  -800., 800. );
  genEtaWithNewZCut       = fs->make<TH1F>( "genEtaWithNewZCut", "Eta of generator particles that satisfy the new Zimpact cut", 100,  -8, 8 );
  genPLossWithOldPtCut        = fs->make<TH1F>( "genPLossWithOldPtCut", "P of generator particles that do not satisfy the old Pt cut", 100,  0., 300. );
  genPtLossWithNewPCut        = fs->make<TH1F>( "genPtLossWithNewPCut", "Pt of generator particles that do not satisfy the new P cut", 100,  0., 50. );
  genZLossWithOldEtaCut       = fs->make<TH1F>( "genZLossWithOldEtaCut", "Zimpact of generator particles that do not satisfy the old Eta cut", 100,  -800., 800. );
  genEtaLossWithNewZCut       = fs->make<TH1F>( "genEtaLossWithNewZCut", "Eta of generator particles that do not satisfy the new Zimpact cut", 100,  -8, 8 );


  genPversusPt        = fs->make<TH2F>( "genPversusPt", "P vs Pt of generator particles", 100,  0., 300.,100,0,50 );
  genZversusEta        = fs->make<TH2F>( "genZversusEta", "Zimpact vs Eta of generator particles", 100,  -800., 800.,100,-8.,8. );

                          
  genChainP               = fs->make<TH1F>( "genChainP"      , "P of particles belonging to decay chain", 100,  0., 300. );
  genChainPt              = fs->make<TH1F>( "genChainPt"      , "Pt of particles belonging to decay chain", 100,  0., 50. );
  genChainEta             = fs->make<TH1F>( "genChainEta"      , "Eta of particles belonging to decay chain", 100,  -8., 8. );
  genChainPID             = fs->make<TH1F>( "genChainPID"    , "PID of particles belonging to decay chain", 100,  0., 1000. );
                          
  genChainPWithOldPtCut   = fs->make<TH1F>( "genChainPWithOldPtCut", "P of generator particles that satisfy the old Pt cut", 100,  0., 300. );
  genChainPtWithNewPCut   = fs->make<TH1F>( "genChainPtWithNewPCut", "Pt of generator particles that satisfy the new P cut", 100,  0., 50. );
  genChainZWithOldEtaCut  = fs->make<TH1F>( "genChainZWithOldEtaCut", "Zimpact of generator particles that satisfy the old Eta cut", 100,  -800., 800. );
  genChainEtaWithNewZCut  = fs->make<TH1F>( "genChainEtaWithNewZCut", "Eta of generator particles that satisfy the new Zimpact cut", 100,  -8, 8 );
}

void
SimTrackSimVertexDumper::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   using namespace HepMC;

   std::vector<SimTrack> theSimTracks;
   std::vector<SimVertex> theSimVertexes;

   Handle<HepMCProduct> MCEvt;
   Handle<SimTrackContainer> SimTk;
   Handle<SimVertexContainer> SimVtx;

   iEvent.getByLabel(HepMCLabel, MCEvt);
   const HepMC::GenEvent* evt = MCEvt->GetEvent();


   iEvent.getByLabel(SimTkLabel,SimTk);
   iEvent.getByLabel(SimVtxLabel,SimVtx);

   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

   std::cout << "\n SimVertex / SimTrack structure dump \n" << std::endl;
   std::cout << " SimVertex in the event = " << theSimVertexes.size() << std::endl;
   std::cout << " SimTracks in the event = " << theSimTracks.size() << std::endl;
   std::cout << "\n" << std::endl;
   for (unsigned int isimvtx = 0; isimvtx < theSimVertexes.size(); isimvtx++){
     std::cout << "SimVertex " << isimvtx << " = " << theSimVertexes[isimvtx] << "\n" << std::endl;
     for (unsigned int isimtk = 0; isimtk < theSimTracks.size() ; isimtk++ ) {
       if ( theSimTracks[isimtk].vertIndex() >= 0 && abs(theSimTracks[isimtk].vertIndex()) == isimvtx ) {
         std::cout<<"  SimTrack " << isimtk << " = "<< theSimTracks[isimtk] 
		  <<" Track Id = "<<theSimTracks[isimtk].trackId()<< std::endl;

         // for debugging purposes
         if (dumpHepMC ) {
           if ( theSimTracks[isimtk].genpartIndex() != -1 ) {
             HepMC::GenParticle* part = evt->barcode_to_particle( theSimTracks[isimtk].genpartIndex() ) ;
             std::cout << "  ---> Corresponding to HepMC particle " << *part << std::endl;
           }
         }
       }
     }
     std::cout << "\n" << std::endl;
   }
   
   for (std::vector<SimTrack>::iterator isimtk = theSimTracks.begin();
        isimtk != theSimTracks.end(); ++isimtk){
     if(isimtk->noVertex()){
       std::cout<<"SimTrack without an associated Vertex = "<< *isimtk <<std::endl;
     }
   }
   
   //Aggiunta per fare analisis 

   float Z_lmax = 2.9*( ( 1 - exp(-2*(5.5)) ) / ( 2*exp(-(5.5)) ) );
   float Z_lmin = 2.9*( ( 1 - exp(-2*(-5.5)) ) / ( 2*exp(-(-5.5)) ) );

   std::cout<<"Z_min = "<<Z_lmin<<"; Z_max = "<<Z_lmax<<std::endl;

   for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
       vitr != evt->vertices_end(); ++vitr ) { 
     // loop for vertex ...
     // real vertex?
     bool qvtx=false;
     
     for (HepMC::GenVertex::particle_iterator pitr= (*vitr)->particles_begin(HepMC::children);
          pitr != (*vitr)->particles_end(HepMC::children); ++pitr) {
       // Admit also status=1 && end_vertex for long vertex special decay treatment 
       if ((*pitr)->status()==1) {
         qvtx=true;
         break;
       }  
       // The selection is made considering if the partcile with status = 2 have the end_vertex
       // with a radius (R) greater then the theRDecLenCut that means: the end_vertex is outside
       // the beampipe cilinder (no requirement on the Z of the vertex is applyed).
       else if ( (*pitr)->status()== 2 ) {
         if ( (*pitr)->end_vertex() != 0  ) { 
           //double xx = x0-(*pitr)->end_vertex()->position().x();
          //double yy = y0-(*pitr)->end_vertex()->position().y();
           double xx = (*pitr)->end_vertex()->position().x();
           double yy = (*pitr)->end_vertex()->position().y();
           double r_dd=std::sqrt(xx*xx+yy*yy);
           if (r_dd>2.9){
             qvtx=true;
             break;
           }
         }
       }
     }
     
     
     if (!qvtx) {
       continue;
     }
     
     double x1 = (*vitr)->position().x();
     double y1 = (*vitr)->position().y();
     double z1 = (*vitr)->position().z();
     
     for (HepMC::GenVertex::particle_iterator vpitr= (*vitr)->particles_begin(HepMC::children);
          vpitr != (*vitr)->particles_end(HepMC::children); ++vpitr){
       double r_decay_length=-1;
       if ( (*vpitr)->status() == 1 || (*vpitr)->status() == 2 ) {
         if ( (*vpitr)->end_vertex() != 0 ) { 
           double x2 = (*vpitr)->end_vertex()->position().x();
           double y2 = (*vpitr)->end_vertex()->position().y();
           r_decay_length=std::sqrt(x2*x2+y2*y2);
         }
       } 
       
       math::XYZTLorentzVector p((*vpitr)->momentum().px(),
                                 (*vpitr)->momentum().py(),
                                 (*vpitr)->momentum().pz(),
                                 (*vpitr)->momentum().e());

       double zimpact = (2.9-sqrt(x1*x1+y1*y1))*(1/tan(p.Theta()))+z1;
       
       if( (*vpitr)->status() == 1 ) {
         genP->Fill(p.P());
         genPt->Fill(p.Pt());
         genPversusPt->Fill(p.P(),p.Pt());
         genEta->Fill(p.Eta());
         genZimpact->Fill(zimpact);
         genZversusEta->Fill(zimpact,p.Eta());
         if(fabs(p.Eta())<5.5)
           genZWithOldEtaCut->Fill(zimpact);
         if(fabs(p.Eta())>5.5)
           genZLossWithOldEtaCut->Fill(zimpact);
         if(zimpact>Z_lmin&&zimpact<Z_lmax)
           genEtaWithNewZCut->Fill(p.Eta());
         if(zimpact<Z_lmin||zimpact>Z_lmax)
           genEtaLossWithNewZCut->Fill(p.Eta());
         if(p.P()>0.04)
           genPtWithNewPCut->Fill(p.Pt());
         if(p.P()<0.04)
           genPtLossWithNewPCut->Fill(p.Pt());
         if(p.Pt()>0.04)
           genPWithOldPtCut->Fill(p.P());
         if(p.Pt()<0.04)
           genPLossWithOldPtCut->Fill(p.P());
         
         genPID->Fill((*vpitr)->pdg_id());
         if(p.P()<0.04||zimpact<Z_lmin||zimpact>Z_lmax)
           genPIDLossNew->Fill((*vpitr)->pdg_id());
         if(p.Pt()<0.04||fabs(p.Eta())>5.5)
           genPIDLossOld->Fill((*vpitr)->pdg_id());

       }
       else if((*vpitr)->status() == 2 && r_decay_length > 2.9){
         genChainP->Fill(p.P());
         genChainPt->Fill(p.Pt());
         genChainEta->Fill(p.Eta());
         genChainPID->Fill((*vpitr)->pdg_id());
         if(fabs(p.Eta())<5.5){
           genChainZWithOldEtaCut->Fill(zimpact);
         }
         if(zimpact>Z_lmin&&zimpact<Z_lmax){
           genChainEtaWithNewZCut->Fill(p.Eta());
         }
         if(p.P()>0.04){
           genChainPtWithNewPCut->Fill(p.Pt());
         }
         if(p.Pt()>0.04){
           genChainPWithOldPtCut->Fill(p.P());
         }
         particleAssignDaughters(*vpitr,Z_lmin,Z_lmax);
       }
     }
   }
   return;
}


void SimTrackSimVertexDumper::particleAssignDaughters( HepMC::GenParticle* vp,float Z_lmin,float Z_lmax)
{
 
  if ( !(vp->end_vertex())  ) return ;
  double x1 = vp->end_vertex()->position().x();
  double y1 = vp->end_vertex()->position().y();
  double z1 = vp->end_vertex()->position().z();  
  math::XYZTLorentzVector p(vp->momentum().px(), vp->momentum().py(), vp->momentum().pz(), vp->momentum().e());
  for (HepMC::GenVertex::particle_iterator 
         vpdec= vp->end_vertex()->particles_begin(HepMC::children);
       vpdec != vp->end_vertex()->particles_end(HepMC::children); ++vpdec) {
    
    //transform decay products such that in the rest frame of mother
    math::XYZTLorentzVector pdec((*vpdec)->momentum().px(),
                                 (*vpdec)->momentum().py(),
                                 (*vpdec)->momentum().pz(),
                                 (*vpdec)->momentum().e());

    double zimpact = (2.9-sqrt(x1*x1+y1*y1))*(1/tan(pdec.Theta()))+z1;

    genChainP->Fill(pdec.P());
    genChainPt->Fill(pdec.Pt());
    genChainEta->Fill(pdec.Eta());
    genChainPID->Fill((*vpdec)->pdg_id());
    if(fabs(p.Eta())<5.5)
      genChainZWithOldEtaCut->Fill(zimpact);
    if(zimpact>Z_lmin&&zimpact<Z_lmax)
      genChainEtaWithNewZCut->Fill(pdec.Eta());
    if(pdec.P()>0.04)
      genChainPtWithNewPCut->Fill(pdec.Pt());
    if(pdec.Pt()>0.04)
      genChainPWithOldPtCut->Fill(pdec.P());
    
    // children should only be taken into account once
    if ( (*vpdec)->status() == 2 && (*vpdec)->end_vertex() != 0 ) 
      {
        particleAssignDaughters(*vpdec,Z_lmin,Z_lmax);
      }
    (*vpdec)->set_status(1000+(*vpdec)->status()); 
  }
  return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(SimTrackSimVertexDumper);
