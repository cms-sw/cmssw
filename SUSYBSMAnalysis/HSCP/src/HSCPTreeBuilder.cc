// -*- C++ -*-
//
// Package:    HSCPTreeBuilder
// Class:      HSCPTreeBuilder
// 
/**\class HSCPTreeBuilder HSCPTreeBuilder.cc SUSYBSMAnalysis/HSCP/src/HSCPTreeBuilder.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Thu Mar 11 12:19:07 CEST 2010
// $Id: HSCPTreeBuilder.cc,v 1.8 2012/12/26 22:12:10 wmtan Exp $
//


#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"

#include <MagneticField/Engine/interface/MagneticField.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"


#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TTree.h"
#include "TROOT.h"

#include <ext/hash_map>


using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;

#define MAX_VERTICES 1000
#define MAX_HSCPS    10000
#define MAX_GENS     10000
#define MAX_ECALCRYS 10

class HSCPTreeBuilder : public edm::EDFilter {
	public:
		explicit HSCPTreeBuilder(const edm::ParameterSet&);
		~HSCPTreeBuilder();


	private:
		virtual void beginJob() ;
		virtual bool filter(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
                int ClosestMuonIndex(reco::TrackRef track, std::vector<reco::MuonRef>);

		const edm::EventSetup* iSetup_;
		const edm::Event*      iEvent_;

		edm::Service<TFileService> tfs;
                InputTag       m_HSCPsTag;
                bool           reccordVertexInfo;
                bool           reccordGenInfo;

		TTree*         MyTree;
		bool           Event_triggerL1Bits[192];
                bool           Event_triggerHLTBits[128];
                bool           Event_technicalBits[64];
		unsigned int   Event_EventNumber;
		unsigned int   Event_RunNumber;
                unsigned int   Event_LumiSection;
                unsigned int   Event_BXCrossing;
                unsigned int   Event_Orbit;
                unsigned int   Event_Store;
                unsigned int   Event_Time;
                bool           Event_PhysicsDeclared;
                float          Event_BField;

                unsigned int   NVertices;
		float	       Vertex_x           [MAX_VERTICES];
                float          Vertex_y           [MAX_VERTICES];
                float          Vertex_z           [MAX_VERTICES];
                float          Vertex_x_err       [MAX_VERTICES];
                float          Vertex_y_err       [MAX_VERTICES];
                float          Vertex_z_err       [MAX_VERTICES];
                int            Vertex_TrackSize   [MAX_VERTICES];
		float	       Vertex_chi2        [MAX_VERTICES];
		float	       Vertex_ndof        [MAX_VERTICES];
                bool           Vertex_isFake      [MAX_VERTICES];

                unsigned int   NHSCPs;
                bool           Hscp_hasTrack      [MAX_HSCPS];
                bool           Hscp_hasMuon       [MAX_HSCPS];
                bool           Hscp_hasRpc        [MAX_HSCPS];
                bool           Hscp_hasCalo       [MAX_HSCPS];
                int            Hscp_type          [MAX_HSCPS];
		unsigned int   Track_NOH          [MAX_HSCPS];
		float          Track_p            [MAX_HSCPS];
		float          Track_pt           [MAX_HSCPS];
                float          Track_pt_err       [MAX_HSCPS];
		float          Track_chi2         [MAX_HSCPS];
                unsigned int   Track_ndof         [MAX_HSCPS];
		float          Track_eta          [MAX_HSCPS];  
                float          Track_eta_err      [MAX_HSCPS];
		float          Track_phi          [MAX_HSCPS];
                float          Track_phi_err      [MAX_HSCPS];
		float          Track_dz           [MAX_HSCPS];
		float          Track_d0           [MAX_HSCPS];
		int            Track_quality      [MAX_HSCPS];
		int            Track_charge       [MAX_HSCPS];
		float          Track_dEdxE1       [MAX_HSCPS];
                float          Track_dEdxE1_NOS   [MAX_HSCPS];
		unsigned int   Track_dEdxE1_NOM   [MAX_HSCPS];
                float          Track_dEdxE2       [MAX_HSCPS];
                float          Track_dEdxE2_NOS   [MAX_HSCPS];
                unsigned int   Track_dEdxE2_NOM   [MAX_HSCPS];
                float          Track_dEdxE3       [MAX_HSCPS];
                float          Track_dEdxE3_NOS   [MAX_HSCPS];
                unsigned int   Track_dEdxE3_NOM   [MAX_HSCPS];
                float          Track_dEdxD1       [MAX_HSCPS];
                float          Track_dEdxD1_NOS   [MAX_HSCPS];
                unsigned int   Track_dEdxD1_NOM   [MAX_HSCPS];
                float          Track_dEdxD2       [MAX_HSCPS]; 
                float          Track_dEdxD2_NOS   [MAX_HSCPS];
                unsigned int   Track_dEdxD2_NOM   [MAX_HSCPS];
                float          Track_dEdxD3       [MAX_HSCPS];
                float          Track_dEdxD3_NOS   [MAX_HSCPS];
                unsigned int   Track_dEdxD3_NOM   [MAX_HSCPS];
                float          Muon_p             [MAX_HSCPS];
                float          Muon_pt            [MAX_HSCPS];
                float          Muon_eta           [MAX_HSCPS];
                float          Muon_phi           [MAX_HSCPS];
                int            Muon_type          [MAX_HSCPS];
                bool           Muon_qualityValid  [MAX_HSCPS];
                int            Muon_charge        [MAX_HSCPS];
                float          Muon_dt_IBeta      [MAX_HSCPS];
                float          Muon_dt_IBeta_err  [MAX_HSCPS];
                float          Muon_dt_fIBeta     [MAX_HSCPS];
                float          Muon_dt_fIBeta_err [MAX_HSCPS];
                int            Muon_dt_ndof       [MAX_HSCPS];
                float          Muon_csc_IBeta     [MAX_HSCPS];
                float          Muon_csc_IBeta_err [MAX_HSCPS];
                float          Muon_csc_fIBeta    [MAX_HSCPS];
                float          Muon_csc_fIBeta_err[MAX_HSCPS];
                int            Muon_csc_ndof      [MAX_HSCPS];
                float          Muon_cb_IBeta      [MAX_HSCPS];
                float          Muon_cb_IBeta_err  [MAX_HSCPS];
                float          Muon_cb_fIBeta     [MAX_HSCPS];
                float          Muon_cb_fIBeta_err [MAX_HSCPS];
                int            Muon_cb_ndof       [MAX_HSCPS];          
                float          Rpc_beta           [MAX_HSCPS];

                float          Calo_ecal_crossedE         [MAX_HSCPS];
                float          Calo_ecal_beta             [MAX_HSCPS];
                float          Calo_ecal_beta_err         [MAX_HSCPS];
                float          Calo_ecal_invBeta_err      [MAX_HSCPS];
                float          Calo_ecal_dEdx             [MAX_HSCPS];
                float          Calo_ecal_time             [MAX_HSCPS];
                float          Calo_ecal_time_err         [MAX_HSCPS];
                int            Calo_ecal_numCrysCrossed   [MAX_HSCPS];
                float          Calo_ecal_swissCrossKs     [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_e1OverE9s        [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_trackLengths     [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_trackExitEtas    [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_trackExitPhis    [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_energies         [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_outOfTimeEnergies[MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_chi2s            [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_outOfTimeChi2s   [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_times            [MAX_HSCPS][MAX_ECALCRYS];
                float          Calo_ecal_timeErrors       [MAX_HSCPS][MAX_ECALCRYS];
                unsigned int   Calo_ecal_detIds           [MAX_HSCPS][MAX_ECALCRYS];

                unsigned int   NGens;
                int            Gen_pdgId          [MAX_GENS];
                float          Gen_charge         [MAX_GENS];
                float          Gen_p              [MAX_GENS];
                float          Gen_px             [MAX_GENS];
                float          Gen_py             [MAX_GENS];
                float          Gen_pz             [MAX_GENS];
                float          Gen_pt             [MAX_GENS];
                float          Gen_eta            [MAX_GENS];
                float          Gen_phi            [MAX_GENS];
                float          Gen_beta           [MAX_GENS];
                float          Gen_mass           [MAX_GENS];
};

HSCPTreeBuilder::HSCPTreeBuilder(const edm::ParameterSet& iConfig)
{
   m_HSCPsTag          = iConfig.getParameter<InputTag>              ("HSCParticles");

   reccordVertexInfo   = iConfig.getUntrackedParameter<bool>    ("reccordVertexInfo"  ,  true );
   reccordGenInfo      = iConfig.getUntrackedParameter<bool>    ("reccordGenInfo"     ,  false );

   std::cout << "######################################################" << endl;
   std::cout << "      USE OF THE HSCPTreeBuilder is deprecated!       " << endl;
   std::cout << "better to use the HSCParticle Producer and then FWLite" << endl;
   std::cout << "######################################################" << endl;

}


HSCPTreeBuilder::~HSCPTreeBuilder()
{
}

void
HSCPTreeBuilder::beginJob()
{
   TTree::SetMaxTreeSize(1000*Long64_t(2000000000)); // authorize Trees up to 2 Terabytes
   MyTree = tfs->make<TTree> ("HscpTree","HscpTree");

   MyTree->Branch("Event_EventNumber"       ,&Event_EventNumber     ,"Event_EventNumber/i");
   MyTree->Branch("Event_RunNumber"         ,&Event_RunNumber       ,"Event_RunNumber/i");
   MyTree->Branch("Event_LumiSection"       ,&Event_LumiSection     ,"Event_LumiSection/i");
   MyTree->Branch("Event_BXCrossing"        ,&Event_BXCrossing      ,"Event_BXCrossing/i");
   MyTree->Branch("Event_Orbit"             ,&Event_Orbit           ,"Event_Orbit/i");
   MyTree->Branch("Event_Store"             ,&Event_Store           ,"Event_Store/i");
   MyTree->Branch("Event_Time"              ,&Event_Time            ,"Event_Time/i");
   MyTree->Branch("Event_PhysicsDeclared"   ,&Event_PhysicsDeclared ,"Event_PhysicsDeclared/O");
   MyTree->Branch("Event_technicalBits"     ,Event_technicalBits    ,"Event_technicalBits[64]/O");
   MyTree->Branch("Event_triggerL1Bits"     ,Event_triggerL1Bits    ,"Event_triggerL1Bits[192]/O");
   MyTree->Branch("Event_triggerHLTBits"    ,Event_triggerHLTBits   ,"Event_triggerHLTBits[128]/O");
   MyTree->Branch("Event_BField"            ,&Event_BField          ,"Event_BField/F");

   if(reccordVertexInfo){
   MyTree->Branch("NVertices"       ,&NVertices      ,"NVertices/I");
   MyTree->Branch("Vertex_x"        ,Vertex_x        ,"Vertex_x[NVertices]/F");
   MyTree->Branch("Vertex_y"        ,Vertex_y        ,"Vertex_y[NVertices]/F");
   MyTree->Branch("Vertex_z"        ,Vertex_z        ,"Vertex_z[NVertices]/F");
   MyTree->Branch("Vertex_x_err"    ,Vertex_x_err    ,"Vertex_x_err[NVertices]/F");
   MyTree->Branch("Vertex_y_err"    ,Vertex_y_err    ,"Vertex_y_err[NVertices]/F");
   MyTree->Branch("Vertex_z_err"    ,Vertex_z_err    ,"Vertex_z_err[NVertices]/F");
   MyTree->Branch("Vertex_TrackSize",Vertex_TrackSize,"Vertex_TrackSize[NVertices]/I");
   MyTree->Branch("Vertex_chi2"     ,Vertex_chi2     ,"Vertex_chi2[NVertices]/F");
   MyTree->Branch("Vertex_ndof"     ,Vertex_ndof     ,"Vertex_ndof[NVertices]/F");
   MyTree->Branch("Vertex_isFake"   ,Vertex_isFake   ,"Vertex_isFake[NVertices]/O");
   }

   MyTree->Branch("NHSCPs"             ,&NHSCPs            ,"NHSCPs/I");
   MyTree->Branch("Hscp_hasTrack"      ,Hscp_hasTrack      ,"Hscp_hasTrack[NHSCPs]/O");
   MyTree->Branch("Hscp_hasMuon"       ,Hscp_hasMuon       ,"Hscp_hasMuon[NHSCPs]/O");
   MyTree->Branch("Hscp_hasRpc"        ,Hscp_hasRpc        ,"Hscp_hasRpc[NHSCPs]/O");
   MyTree->Branch("Hscp_hasCalo"       ,Hscp_hasCalo       ,"Hscp_hasCalo[NHSCPs]/O");
   MyTree->Branch("Hscp_type"          ,Hscp_type          ,"Hscp_type[NHSCPs]/I");
   MyTree->Branch("Track_NOH"          ,Track_NOH          ,"Track_NOH[NHSCPs]/I");
   MyTree->Branch("Track_p"            ,Track_p            ,"Track_p[NHSCPs]/F");
   MyTree->Branch("Track_pt"           ,Track_pt           ,"Track_pt[NHSCPs]/F");
   MyTree->Branch("Track_pt_err"       ,Track_pt_err       ,"Track_pt_err[NHSCPs]/F");
   MyTree->Branch("Track_chi2"         ,Track_chi2         ,"Track_chi2[NHSCPs]/F");
   MyTree->Branch("Track_ndof"         ,Track_ndof         ,"Track_ndof[NHSCPs]/F");
   MyTree->Branch("Track_eta"          ,Track_eta          ,"Track_eta[NHSCPs]/F");
   MyTree->Branch("Track_eta_err"      ,Track_eta_err      ,"Track_eta_err[NHSCPs]/F");
   MyTree->Branch("Track_phi"          ,Track_phi          ,"Track_phi[NHSCPs]/F");
   MyTree->Branch("Track_phi_err"      ,Track_phi_err      ,"Track_phi_err[NHSCPs]/F");
   MyTree->Branch("Track_d0"           ,Track_d0           ,"Track_d0[NHSCPs]/F");
   MyTree->Branch("Track_dz"           ,Track_dz           ,"Track_dz[NHSCPs]/F");
   MyTree->Branch("Track_quality"      ,Track_quality      ,"Track_quality[NHSCPs]/I");
   MyTree->Branch("Track_charge"       ,Track_charge       ,"Track_charge[NHSCPs]/I");
   MyTree->Branch("Track_dEdxE1"       ,Track_dEdxE1       ,"Track_dEdxE1[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE1_NOS"   ,Track_dEdxE1_NOS   ,"Track_dEdxE1_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE1_NOM"   ,Track_dEdxE1_NOM   ,"Track_dEdxE1_NOM[NHSCPs]/I");
   MyTree->Branch("Track_dEdxE2"       ,Track_dEdxE2       ,"Track_dEdxE2[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE2_NOS"   ,Track_dEdxE2_NOS   ,"Track_dEdxE2_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE2_NOM"   ,Track_dEdxE2_NOM   ,"Track_dEdxE2_NOM[NHSCPs]/I");
   MyTree->Branch("Track_dEdxE3"       ,Track_dEdxE3       ,"Track_dEdxE3[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE3_NOS"   ,Track_dEdxE3_NOS   ,"Track_dEdxE3_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxE3_NOM"   ,Track_dEdxE3_NOM   ,"Track_dEdxE3_NOM[NHSCPs]/I");
   MyTree->Branch("Track_dEdxD1"       ,Track_dEdxD1       ,"Track_dEdxD1[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD1_NOS"   ,Track_dEdxD1_NOS   ,"Track_dEdxD1_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD1_NOM"   ,Track_dEdxD1_NOM   ,"Track_dEdxD1_NOM[NHSCPs]/I");
   MyTree->Branch("Track_dEdxD2"       ,Track_dEdxD2       ,"Track_dEdxD2[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD2_NOS"   ,Track_dEdxD2_NOS   ,"Track_dEdxD2_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD2_NOM"   ,Track_dEdxD2_NOM   ,"Track_dEdxD2_NOM[NHSCPs]/I");
   MyTree->Branch("Track_dEdxD3"       ,Track_dEdxD3       ,"Track_dEdxD3[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD3_NOS"   ,Track_dEdxD3_NOS   ,"Track_dEdxD3_NOS[NHSCPs]/F");
   MyTree->Branch("Track_dEdxD3_NOM"   ,Track_dEdxD3_NOM   ,"Track_dEdxD3_NOM[NHSCPs]/I");
   MyTree->Branch("Muon_p"             ,Muon_p             ,"Muon_p[NHSCPs]/F");
   MyTree->Branch("Muon_pt"            ,Muon_pt            ,"Muon_pt[NHSCPs]/F");
   MyTree->Branch("Muon_eta"           ,Muon_eta           ,"Muon_eta[NHSCPs]/F");
   MyTree->Branch("Muon_phi"           ,Muon_phi           ,"Muon_phi[NHSCPs]/F");
   MyTree->Branch("Muon_type"          ,Muon_type          ,"Muon_type[NHSCPs]/i");
   MyTree->Branch("Muon_qualityValid"  ,Muon_qualityValid  ,"Muon_qualityValid[NHSCPs]/O");
   MyTree->Branch("Muon_charge"        ,Muon_charge        ,"Muon_charge[NHSCPs]/i");
   MyTree->Branch("Muon_dt_IBeta"      ,Muon_dt_IBeta      ,"Muon_dt_IBeta[NHSCPs]/F");
   MyTree->Branch("Muon_dt_IBeta_err"  ,Muon_dt_IBeta_err  ,"Muon_dt_IBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_dt_fIBeta"     ,Muon_dt_fIBeta     ,"Muon_dt_fIBeta[NHSCPs]/F");
   MyTree->Branch("Muon_dt_fIBeta_err" ,Muon_dt_fIBeta_err ,"Muon_dt_fIBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_dt_ndof"       ,Muon_dt_ndof       ,"Muon_dt_ndof[NHSCPs]/I");
   MyTree->Branch("Muon_csc_IBeta"     ,Muon_csc_IBeta     ,"Muon_csc_IBeta[NHSCPs]/F");
   MyTree->Branch("Muon_csc_IBeta_err" ,Muon_csc_IBeta_err ,"Muon_csc_IBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_csc_fIBeta"    ,Muon_csc_fIBeta    ,"Muon_csc_fIBeta[NHSCPs]/F");
   MyTree->Branch("Muon_csc_fIBeta_err",Muon_csc_fIBeta_err,"Muon_csc_fIBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_csc_ndof"      ,Muon_csc_ndof      ,"Muon_csc_ndof[NHSCPs]/I");
   MyTree->Branch("Muon_cb_IBeta"      ,Muon_cb_IBeta      ,"Muon_cb_IBeta[NHSCPs]/F");
   MyTree->Branch("Muon_cb_IBeta_err"  ,Muon_cb_IBeta_err  ,"Muon_cb_IBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_cb_fIBeta"     ,Muon_cb_fIBeta     ,"Muon_cb_fIBeta[NHSCPs]/F");
   MyTree->Branch("Muon_cb_fIBeta_err" ,Muon_cb_fIBeta_err ,"Muon_cb_fIBeta_err[NHSCPs]/F");
   MyTree->Branch("Muon_cb_ndof"       ,Muon_cb_ndof       ,"Muon_cb_ndof[NHSCPs]/I");

   MyTree->Branch("Rpc_beta"           ,Rpc_beta           ,"Rpc_beta[NHSCPs]/F");

   MyTree->Branch("Calo_ecal_crossedE"         ,Calo_ecal_crossedE      ,"Calo_ecal_crossedE[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_beta"             ,Calo_ecal_beta          ,"Calo_ecal_beta[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_beta_err"         ,Calo_ecal_beta_err      ,"Calo_ecal_beta_err[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_invBeta_err"      ,Calo_ecal_invBeta_err   ,"Calo_ecal_invBeta_err[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_dEdx"             ,Calo_ecal_dEdx          ,"Calo_ecal_dEdx[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_time"             ,Calo_ecal_time          ,"Calo_ecal_time[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_time_err"         ,Calo_ecal_time_err      ,"Calo_ecal_time_err[NHSCPs]/F");
   MyTree->Branch("Calo_ecal_numCrysCrossed"   ,Calo_ecal_numCrysCrossed,"Calo_ecal_numCrysCrossed[NHSCPs]/I");
   MyTree->Branch("Calo_ecal_swissCrossKs"     ,Calo_ecal_swissCrossKs  ,"Calo_ecal_swissCrossKs[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_e1OverE9s"        ,Calo_ecal_e1OverE9s     ,"Calo_ecal_e1OverE9s[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_trackLengths"     ,Calo_ecal_trackLengths  ,"Calo_ecal_trackLengths[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_trackExitEtas"    ,Calo_ecal_trackExitEtas ,"Calo_ecal_trackExitEtas[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_trackExitPhis"    ,Calo_ecal_trackExitPhis ,"Calo_ecal_trackExitPhis[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_energies"         ,Calo_ecal_energies      ,"Calo_ecal_energies[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_outOfTimeEnergies",Calo_ecal_outOfTimeEnergies      ,"Calo_ecal_outOfTimeEnergies[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_chi2s"            ,Calo_ecal_chi2s      ,"Calo_ecal_chi2s[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_outOfTimeChi2s"   ,Calo_ecal_outOfTimeChi2s      ,"Calo_ecal_outOfTimeChi2s[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_times"            ,Calo_ecal_times         ,"Calo_ecal_times[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_timeErrors"       ,Calo_ecal_timeErrors    ,"Calo_ecal_timeErrors[NHSCPs][10]/F");
   MyTree->Branch("Calo_ecal_detIds"           ,Calo_ecal_detIds        ,"Calo_ecal_detIds[NHSCPs][10]/I");

   if(reccordGenInfo){
   MyTree->Branch("NGens"              ,&NGens             ,"NGens/I");
   MyTree->Branch("Gen_pdgId"          ,Gen_pdgId          ,"Gen_pdgId[NGens]/i");
   MyTree->Branch("Gen_charge"         ,Gen_charge         ,"Gen_charge[NGens]/F");
   MyTree->Branch("Gen_p"              ,Gen_p              ,"Gen_p[NGens]/F");
   MyTree->Branch("Gen_px"             ,Gen_px             ,"Gen_px[NGens]/F");
   MyTree->Branch("Gen_py"             ,Gen_py             ,"Gen_py[NGens]/F");
   MyTree->Branch("Gen_pz"             ,Gen_pz             ,"Gen_pz[NGens]/F");
   MyTree->Branch("Gen_pt"             ,Gen_pt             ,"Gen_pt[NGens]/F");
   MyTree->Branch("Gen_eta"            ,Gen_eta            ,"Gen_eta[NGens]/F");
   MyTree->Branch("Gen_phi"            ,Gen_phi            ,"Gen_phi[NGens]/F");
   MyTree->Branch("Gen_beta"           ,Gen_beta           ,"Gen_beta[NGens]/F");
   MyTree->Branch("Gen_mass"           ,Gen_mass           ,"Gen_mass[NGens]/F");
   }

}

void
HSCPTreeBuilder::endJob() 
{
}



bool
HSCPTreeBuilder::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   bool debug = false;
   if (debug) cout << "I'm in HSCPTreeBuilder::analyze!" << endl;


   Event_EventNumber     = iEvent.id().event();
   Event_RunNumber       = iEvent.id().run();
   Event_LumiSection     = iEvent.eventAuxiliary().luminosityBlock();
   Event_BXCrossing      = iEvent.eventAuxiliary().bunchCrossing();
   Event_Orbit           = iEvent.eventAuxiliary().orbitNumber();
   Event_Store           = iEvent.eventAuxiliary().storeNumber();
   Event_Time            = iEvent.eventAuxiliary().time().value();

   // BField part:
   ESHandle<MagneticField> MF;
   iSetup.get<IdealMagneticFieldRecord>().get(MF);
   const MagneticField* theMagneticField = MF.product();
   Event_BField = fabs(theMagneticField->inTesla(GlobalPoint(0,0,0)).z());

   // L1 TRIGGER part:
   edm::Handle<L1GlobalTriggerReadoutRecord> h_gtReadoutRecord;
   iEvent.getByLabel("gtDigis", h_gtReadoutRecord);
   L1GtFdlWord fdlWord = h_gtReadoutRecord->gtFdlWord();
   TechnicalTriggerWord L1technical = fdlWord.gtTechnicalTriggerWord();
   Event_PhysicsDeclared = h_gtReadoutRecord->gtFdlWord().physicsDeclared();
   for (unsigned int i = 0; i <  64; ++i){ Event_technicalBits [i] = L1technical[i]; }
   DecisionWord L1decision = fdlWord.gtDecisionWord();
   for (unsigned int i = 0; i < 128; ++i){ Event_triggerL1Bits [i] = L1decision[i]; }
   DecisionWordExtended L1decisionE = fdlWord.gtDecisionWordExtended();
   for (unsigned int i = 0; i < 64; ++i){ Event_triggerL1Bits [128+i] = L1decisionE[i]; }

   // HLT TRIGGER part:
   edm::Handle<edm::TriggerResults> trh;
   iEvent.getByLabel("TriggerResults", trh);
   for(unsigned int i=0;i<trh->size() && i<128;++i){Event_triggerHLTBits[i] = trh->at(i).accept();}

   edm::Handle<reco::VertexCollection> recoVertexHandle;
   iEvent.getByLabel("offlinePrimaryVertices", recoVertexHandle);
   reco::VertexCollection recoVertex = *recoVertexHandle;

   if(reccordVertexInfo){
   NVertices = 0;
   for(unsigned int i=0;i<recoVertex.size();i++){
      Vertex_x        [NVertices] = recoVertex[i].x();
      Vertex_y        [NVertices] = recoVertex[i].y();
      Vertex_z        [NVertices] = recoVertex[i].z();
      Vertex_x_err    [NVertices] = recoVertex[i].xError();
      Vertex_y_err    [NVertices] = recoVertex[i].yError();
      Vertex_z_err    [NVertices] = recoVertex[i].zError();
      Vertex_TrackSize[NVertices] = recoVertex[i].tracksSize();
      Vertex_chi2     [NVertices] = recoVertex[i].chi2();
      Vertex_ndof     [NVertices] = recoVertex[i].ndof();
      Vertex_isFake   [NVertices] = recoVertex[i].isFake();
      NVertices++; 
   }
   }

   // Source Collection
   edm::Handle<susybsm::HSCParticleCollection > HSCPCollectionHandle;
   iEvent.getByLabel(m_HSCPsTag, HSCPCollectionHandle);
   susybsm::HSCParticleCollection HSCPCollection = *HSCPCollectionHandle.product();

   NHSCPs=0;
   for(unsigned int i=0; i<HSCPCollection.size();i++){
      susybsm::HSCParticle hscp = HSCPCollection[i]; 
      reco::MuonRef  muon  = hscp.muonRef();
      reco::TrackRef track = hscp.trackRef();;

       Hscp_hasTrack        [NHSCPs] = hscp.hasTrackRef();
       Hscp_hasMuon         [NHSCPs] = hscp.hasMuonRef();
       Hscp_hasRpc          [NHSCPs] = hscp.hasRpcInfo();
       Hscp_hasCalo         [NHSCPs] = hscp.hasCaloInfo();
       Hscp_type            [NHSCPs] = hscp.type();

      if(track.isNonnull() && Hscp_hasTrack){
         Track_p            [NHSCPs] = track->p();
         Track_pt           [NHSCPs] = track->pt();
         Track_pt_err       [NHSCPs] = track->ptError();
         Track_eta          [NHSCPs] = track->eta();
         Track_eta_err      [NHSCPs] = track->etaError();
         Track_phi          [NHSCPs] = track->phi();
         Track_phi_err      [NHSCPs] = track->phiError();
         Track_NOH          [NHSCPs] = track->found();
         Track_chi2         [NHSCPs] = track->chi2();
         Track_ndof         [NHSCPs] = track->ndof();
         Track_d0           [NHSCPs] = -1.0f * track->dxy(recoVertex[0].position());
         Track_dz           [NHSCPs] = -1.0f * track->dz (recoVertex[0].position());
         Track_quality      [NHSCPs] = track->qualityMask();
         Track_charge       [NHSCPs] = track->charge(); 
/*         Track_dEdxE1       [NHSCPs] = hscp.dedxEstimator1().dEdx();
         Track_dEdxE1_NOM   [NHSCPs] = hscp.dedxEstimator1().numberOfMeasurements();
         Track_dEdxE1_NOS   [NHSCPs] = hscp.dedxEstimator1().numberOfSaturatedMeasurements();
         Track_dEdxE2       [NHSCPs] = hscp.dedxEstimator2().dEdx();
         Track_dEdxE2_NOM   [NHSCPs] = hscp.dedxEstimator2().numberOfMeasurements();
         Track_dEdxE2_NOS   [NHSCPs] = hscp.dedxEstimator2().numberOfSaturatedMeasurements();
         Track_dEdxE3       [NHSCPs] = hscp.dedxEstimator3().dEdx();
         Track_dEdxE3_NOM   [NHSCPs] = hscp.dedxEstimator3().numberOfMeasurements();
         Track_dEdxE3_NOS   [NHSCPs] = hscp.dedxEstimator3().numberOfSaturatedMeasurements();
         Track_dEdxD1       [NHSCPs] = hscp.dedxDiscriminator1().dEdx();
         Track_dEdxD1_NOM   [NHSCPs] = hscp.dedxDiscriminator1().numberOfMeasurements();
         Track_dEdxD1_NOS   [NHSCPs] = hscp.dedxDiscriminator1().numberOfSaturatedMeasurements();
         Track_dEdxD2       [NHSCPs] = hscp.dedxDiscriminator2().dEdx();
         Track_dEdxD2_NOM   [NHSCPs] = hscp.dedxDiscriminator2().numberOfMeasurements();
         Track_dEdxD2_NOS   [NHSCPs] = hscp.dedxDiscriminator2().numberOfSaturatedMeasurements();
         Track_dEdxD3       [NHSCPs] = hscp.dedxDiscriminator3().dEdx();
         Track_dEdxD3_NOM   [NHSCPs] = hscp.dedxDiscriminator3().numberOfMeasurements();
         Track_dEdxD3_NOS   [NHSCPs] = hscp.dedxDiscriminator3().numberOfSaturatedMeasurements();
*/
      }

      if(muon.isNonnull() && Hscp_hasMuon){
         Muon_p             [NHSCPs] = muon->p();
         Muon_pt            [NHSCPs] = muon->pt();
         Muon_eta           [NHSCPs] = muon->eta();
         Muon_phi           [NHSCPs] = muon->phi();
         Muon_type          [NHSCPs] = muon->type();
         Muon_qualityValid  [NHSCPs] = muon->isQualityValid();
         Muon_charge        [NHSCPs] = muon->charge();
/*         Muon_dt_IBeta      [NHSCPs] = hscp.muonTimeDt().inverseBeta();
         Muon_dt_IBeta_err  [NHSCPs] = hscp.muonTimeDt().inverseBetaErr();
         Muon_dt_fIBeta     [NHSCPs] = hscp.muonTimeDt().freeInverseBeta();
         Muon_dt_fIBeta_err [NHSCPs] = hscp.muonTimeDt().freeInverseBetaErr();
         Muon_dt_ndof       [NHSCPs] = hscp.muonTimeDt().nDof();
         Muon_csc_IBeta     [NHSCPs] = hscp.muonTimeCsc().inverseBeta();
         Muon_csc_IBeta_err [NHSCPs] = hscp.muonTimeCsc().inverseBetaErr();
         Muon_csc_fIBeta    [NHSCPs] = hscp.muonTimeCsc().freeInverseBeta();
         Muon_csc_fIBeta_err[NHSCPs] = hscp.muonTimeCsc().freeInverseBetaErr();
         Muon_csc_ndof      [NHSCPs] = hscp.muonTimeCsc().nDof();
         Muon_cb_IBeta      [NHSCPs] = hscp.muonTimeCombined().inverseBeta();
         Muon_cb_IBeta_err  [NHSCPs] = hscp.muonTimeCombined().inverseBetaErr();
         Muon_cb_fIBeta     [NHSCPs] = hscp.muonTimeCombined().freeInverseBeta();
         Muon_cb_fIBeta_err [NHSCPs] = hscp.muonTimeCombined().freeInverseBetaErr();
         Muon_cb_ndof       [NHSCPs] = hscp.muonTimeCombined().nDof();
*/
      }

      if(hscp.hasCaloInfo()){
//         Calo_ecal_crossedE      [NHSCPs] = hscp.calo().ecalCrossedEnergy;
//         Calo_ecal_beta          [NHSCPs] = hscp.calo().ecalBeta;
//         Calo_ecal_beta_err      [NHSCPs] = hscp.calo().ecalBetaError;
//         Calo_ecal_invBeta_err   [NHSCPs] = hscp.calo().ecalInvBetaError;
//         Calo_ecal_dEdx          [NHSCPs] = hscp.calo().ecalDeDx;
//         Calo_ecal_time          [NHSCPs] = hscp.calo().ecalTime;
//         Calo_ecal_time_err      [NHSCPs] = hscp.calo().ecalTimeError;
//         Calo_ecal_numCrysCrossed[NHSCPs] = hscp.calo().ecalCrysCrossed;
/*         for(int i=0; i < Calo_ecal_numCrysCrossed[NHSCPs] && i < MAX_ECALCRYS; ++i)
         {
           Calo_ecal_swissCrossKs     [NHSCPs][i] = hscp.calo().ecalSwissCrossKs[i];
           Calo_ecal_e1OverE9s        [NHSCPs][i] = hscp.calo().ecalE1OverE9s[i];
           Calo_ecal_trackLengths     [NHSCPs][i] = hscp.calo().ecalTrackLengths[i];
           GlobalPoint exitPosition = hscp.calo().ecalTrackExitPositions[i];
           Calo_ecal_trackExitEtas    [NHSCPs][i] = exitPosition.eta();
           Calo_ecal_trackExitPhis    [NHSCPs][i] = exitPosition.phi();
           Calo_ecal_energies         [NHSCPs][i] = hscp.calo().ecalEnergies[i];
           Calo_ecal_outOfTimeEnergies[NHSCPs][i] = hscp.calo().ecalOutOfTimeEnergies[i];
           Calo_ecal_chi2s            [NHSCPs][i] = hscp.calo().ecalChi2s[i];
           Calo_ecal_outOfTimeChi2s   [NHSCPs][i] = hscp.calo().ecalOutOfTimeChi2s[i];
           Calo_ecal_times            [NHSCPs][i] = hscp.calo().ecalTimes[i];
           Calo_ecal_timeErrors       [NHSCPs][i] = hscp.calo().ecalTimeErrors[i];
           Calo_ecal_detIds           [NHSCPs][i] = hscp.calo().ecalDetIds[i];
         }
*/
      }

      if(Hscp_hasRpc){
         Rpc_beta           [NHSCPs] = hscp.rpc().beta;
      }

      NHSCPs++;
   }
   

   if(reccordGenInfo){
   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   NGens=0;
   for(unsigned int i=0;i<genParticles->size();i++){
     const GenParticle & part = (*genParticles)[i];
     if(part.status()!=1)continue;
     if(part.pt()<5)continue;
//     if(fabs(part.pdgId())<1000000) continue;

        Gen_pdgId [NGens]     = part.pdgId();
        Gen_charge[NGens]     = part.charge();
        Gen_p     [NGens]     = part.p();
        Gen_px    [NGens]     = part.px();
        Gen_py    [NGens]     = part.py();
        Gen_pz    [NGens]     = part.pz();
        Gen_pt    [NGens]     = part.pt();
        Gen_eta   [NGens]     = part.eta();
        Gen_phi   [NGens]     = part.phi();
        Gen_beta  [NGens]     = part.p()/part.energy();
        Gen_mass  [NGens]     = part.mass();
        NGens++;
     }     
  }
   

   MyTree->Fill();
   return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPTreeBuilder);










