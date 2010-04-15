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
// $Id: HSCPTreeBuilder.cc,v 1.1 2010/04/14 13:05:03 querten Exp $
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
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
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
#define MAX_TRACKS   10000
#define MAX_MUONS    10000
#define MAX_GENS     10000

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

		std::vector<InputTag> m_dEdxDiscrimTag;
		InputTag     m_tracksTag;
                InputTag     m_muonsTag;
                InputTag     m_muontimingTag;



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
		float	       Vertex_x         [MAX_VERTICES];
                float          Vertex_y         [MAX_VERTICES];
                float          Vertex_z         [MAX_VERTICES];
                float          Vertex_x_err     [MAX_VERTICES];
                float          Vertex_y_err     [MAX_VERTICES];
                float          Vertex_z_err     [MAX_VERTICES];
                int            Vertex_TrackSize [MAX_VERTICES];
		float	       Vertex_chi2      [MAX_VERTICES];
		float	       Vertex_ndof      [MAX_VERTICES];
                bool           Vertex_isFake    [MAX_VERTICES];

                unsigned int   NTracks;
		float**        Track_dEdx;
                float**        Track_dEdx_NOS;
		unsigned int** Track_dEdx_NOM;
		unsigned int   Track_NOH        [MAX_TRACKS];
		float          Track_p          [MAX_TRACKS];
                float          Track_px         [MAX_TRACKS];
                float          Track_py         [MAX_TRACKS];
                float          Track_pz         [MAX_TRACKS];
		float          Track_pt         [MAX_TRACKS];
                float          Track_pt_err     [MAX_TRACKS];
		float          Track_chi2       [MAX_TRACKS];
                unsigned int   Track_ndof       [MAX_TRACKS];
		float          Track_eta        [MAX_TRACKS];  
                float          Track_eta_err    [MAX_TRACKS];
		float          Track_phi        [MAX_TRACKS];
                float          Track_phi_err    [MAX_TRACKS];
		float          Track_theta      [MAX_TRACKS];
		float          Track_dz         [MAX_TRACKS];
		float          Track_d0         [MAX_TRACKS];
		int            Track_quality    [MAX_TRACKS];
		int            Track_charge     [MAX_TRACKS];
                int            Track_MuonIndex  [MAX_TRACKS];
                float          Track_MuonDist   [MAX_TRACKS];

                unsigned int   NMuons;
                float          Muon_p             [MAX_MUONS];
                float          Muon_px            [MAX_MUONS];
                float          Muon_py            [MAX_MUONS];
                float          Muon_pz            [MAX_MUONS];
                float          Muon_pt            [MAX_MUONS];
                float          Muon_chi2          [MAX_MUONS];
                unsigned int   Muon_ndof          [MAX_MUONS];
                float          Muon_eta           [MAX_MUONS];
                float          Muon_phi           [MAX_MUONS];
                float          Muon_theta         [MAX_MUONS];
                int            Muon_type          [MAX_MUONS];
                int            Muon_quality       [MAX_MUONS];
                bool           Muon_qualityValid  [MAX_MUONS];
                int            Muon_charge        [MAX_MUONS];
                float          Muon_dt_IBeta      [MAX_MUONS];
                float          Muon_dt_IBeta_err  [MAX_MUONS];
                float          Muon_dt_fIBeta     [MAX_MUONS];
                float          Muon_dt_fIBeta_err [MAX_MUONS];
                float          Muon_csc_IBeta     [MAX_MUONS];
                float          Muon_csc_IBeta_err [MAX_MUONS];
                float          Muon_csc_fIBeta    [MAX_MUONS];
                float          Muon_csc_fIBeta_err[MAX_MUONS];
                float          Muon_cb_IBeta      [MAX_MUONS];
                float          Muon_cb_IBeta_err  [MAX_MUONS];
                float          Muon_cb_fIBeta     [MAX_MUONS];
                float          Muon_cb_fIBeta_err [MAX_MUONS];
                int            Muon_TrackIndex    [MAX_MUONS];
                float          Muon_Track_pt      [MAX_MUONS];
                float          Muon_Track_eta     [MAX_MUONS];
                float          Muon_Track_phi     [MAX_MUONS];


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


		double        MinTrackMomentum;
		double        MaxTrackMomentum;
		double        MinTrackTMomentum;
		double        MaxTrackTMomentum;
		double        MinTrackEta;
		double        MaxTrackEta;
		int           MinTrackNHits;
		int           MaxTrackNHits;

                bool          reccordVertexInfo;
                bool          reccordTrackInfo;
                bool          reccordMuonInfo;
                bool          reccordGenInfo;
};

HSCPTreeBuilder::HSCPTreeBuilder(const edm::ParameterSet& iConfig)
{
   m_tracksTag         = iConfig.getParameter<InputTag>              ("tracks");
   m_dEdxDiscrimTag    = iConfig.getParameter<std::vector<InputTag> >("dEdxDiscrim");  
   m_muonsTag          = iConfig.getParameter<InputTag>              ("muons");
   m_muontimingTag     = iConfig.getParameter<InputTag>              ("muontiming");


   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  1.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0);
   MinTrackTMomentum   = iConfig.getUntrackedParameter<double>  ("minTrackTMomentum"  ,  1.0);
   MaxTrackTMomentum   = iConfig.getUntrackedParameter<double>  ("maxTrackTMomentum"  ,  99999.0);
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MinTrackNHits       = iConfig.getUntrackedParameter<int>     ("minTrackNHits"      ,  0  );
   MaxTrackNHits       = iConfig.getUntrackedParameter<int>     ("maxTrackNHits"      ,  50 );

   reccordVertexInfo   = iConfig.getUntrackedParameter<bool>    ("reccordVertexInfo"  ,  true );
   reccordTrackInfo    = iConfig.getUntrackedParameter<bool>    ("reccordTrackInfo"   ,  true );
   reccordMuonInfo     = iConfig.getUntrackedParameter<bool>    ("reccordMuonInfo"    ,  true );
   reccordGenInfo      = iConfig.getUntrackedParameter<bool>    ("reccordGenInfo"     ,  false );
}


HSCPTreeBuilder::~HSCPTreeBuilder()
{
}

void
HSCPTreeBuilder::beginJob()
{
   Track_dEdx     = new float       *[m_dEdxDiscrimTag.size()];
   Track_dEdx_NOS = new float       *[m_dEdxDiscrimTag.size()];
   Track_dEdx_NOM = new unsigned int*[m_dEdxDiscrimTag.size()];
   for(unsigned int i=0;i<m_dEdxDiscrimTag.size();i++){
      Track_dEdx    [i] = new float       [MAX_TRACKS];
      Track_dEdx_NOS[i] = new float       [MAX_TRACKS];
      Track_dEdx_NOM[i] = new unsigned int[MAX_TRACKS];
   }

   TTree::SetMaxTreeSize(1000*Long64_t(2000000000)); // authorize Trees up to 2 Terabytes
   MyTree = tfs->make<TTree> ("MyTree","MyTree");

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


   if(reccordTrackInfo){
   MyTree->Branch("NTracks"         ,&NTracks        ,"NTracks/I");
   MyTree->Branch("Track_NOH"       ,Track_NOH       ,"Track_NOH[NTracks]/I");
   MyTree->Branch("Track_p"         ,Track_p         ,"Track_p[NTracks]/F");
   MyTree->Branch("Track_px"        ,Track_px        ,"Track_px[NTracks]/F");
   MyTree->Branch("Track_py"        ,Track_py        ,"Track_py[NTracks]/F");
   MyTree->Branch("Track_pz"        ,Track_pz        ,"Track_pz[NTracks]/F");
   MyTree->Branch("Track_pt"        ,Track_pt        ,"Track_pt[NTracks]/F");
   MyTree->Branch("Track_pt_err"    ,Track_pt_err    ,"Track_pt_err[NTracks]/F");
   MyTree->Branch("Track_chi2"      ,Track_chi2      ,"Track_chi2[NTracks]/F");
   MyTree->Branch("Track_ndof"      ,Track_ndof      ,"Track_ndof[NTracks]/F");
   MyTree->Branch("Track_eta"       ,Track_eta       ,"Track_eta[NTracks]/F");
   MyTree->Branch("Track_eta_err"   ,Track_eta_err   ,"Track_eta_err[NTracks]/F");
   MyTree->Branch("Track_phi"       ,Track_phi       ,"Track_phi[NTracks]/F");
   MyTree->Branch("Track_phi_err"   ,Track_phi_err   ,"Track_phi_err[NTracks]/F");
   MyTree->Branch("Track_theta"     ,Track_theta     ,"Track_theta[NTracks]/F");
   MyTree->Branch("Track_d0"        ,Track_d0        ,"Track_d0[NTracks]/F");
   MyTree->Branch("Track_dz"        ,Track_dz        ,"Track_dz[NTracks]/F");
   MyTree->Branch("Track_quality"   ,Track_quality   ,"Track_quality[NTracks]/I");
   MyTree->Branch("Track_charge"    ,Track_charge    ,"Track_charge[NTracks]/I");
   MyTree->Branch("Track_MuonIndex" ,Track_MuonIndex ,"Track_MuonIndex[NTracks]/i");
   MyTree->Branch("Track_MuonDist"  ,Track_MuonDist  ,"Track_MuonDist[NTracks]/F");

   for(unsigned int i=0;i<m_dEdxDiscrimTag.size();i++){
      char name[1024];
      char type[1024];
      sprintf(name,"Track_dEdx_%s"               ,m_dEdxDiscrimTag[i].encode().c_str());
      sprintf(type,"Track_dEdx_%s[NTracks]/F"    ,m_dEdxDiscrimTag[i].encode().c_str());
      MyTree->Branch(name, Track_dEdx[i], type);

      sprintf(name,"Track_dEdx_%s_NOS"           ,m_dEdxDiscrimTag[i].encode().c_str());
      sprintf(type,"Track_dEdx_%s_NOS[NTracks]/F",m_dEdxDiscrimTag[i].encode().c_str());
      MyTree->Branch(name, Track_dEdx_NOS[i], type);

      sprintf(name,"Track_dEdx_%s_NOM"           ,m_dEdxDiscrimTag[i].encode().c_str());
      sprintf(type,"Track_dEdx_%s_NOM[NTracks]/I",m_dEdxDiscrimTag[i].encode().c_str());
      MyTree->Branch(name, Track_dEdx_NOM[i], type);
   }
   }

   if(reccordMuonInfo){
   MyTree->Branch("NMuons"             ,&NMuons            ,"NMuons/I");
   MyTree->Branch("Muon_p"             ,Muon_p             ,"Muon_p[NMuons]/F");
   MyTree->Branch("Muon_px"            ,Muon_px            ,"Muon_px[NMuons]/F");
   MyTree->Branch("Muon_py"            ,Muon_py            ,"Muon_py[NMuons]/F");
   MyTree->Branch("Muon_pz"            ,Muon_pz            ,"Muon_pz[NMuons]/F");
   MyTree->Branch("Muon_pt"            ,Muon_pt            ,"Muon_pt[NMuons]/F");
   MyTree->Branch("Muon_chi2"          ,Muon_chi2          ,"Muon_chi2[NMuons]/F");
   MyTree->Branch("Muon_ndof"          ,Muon_ndof          ,"Muon_ndof[NMuons]/I");
   MyTree->Branch("Muon_eta"           ,Muon_eta           ,"Muon_eta[NMuons]/F");
   MyTree->Branch("Muon_phi"           ,Muon_phi           ,"Muon_phi[NMuons]/F");
   MyTree->Branch("Muon_theta"         ,Muon_theta         ,"Muon_theta[NMuons]/F");
   MyTree->Branch("Muon_type"          ,Muon_type          ,"Muon_type[NMuons]/i");
   MyTree->Branch("Muon_quality"       ,Muon_quality       ,"Muon_quality[NMuons]/i");
   MyTree->Branch("Muon_qualityValid"  ,Muon_qualityValid  ,"Muon_qualityValid[NMuons]/O");
   MyTree->Branch("Muon_charge"        ,Muon_charge        ,"Muon_charge[NMuons]/i");
   MyTree->Branch("Muon_dt_IBeta"      ,Muon_dt_IBeta      ,"Muon_dt_IBeta[NMuons]/F");
   MyTree->Branch("Muon_dt_IBeta_err"  ,Muon_dt_IBeta_err  ,"Muon_dt_IBeta_err[NMuons]/F");
   MyTree->Branch("Muon_dt_fIBeta"     ,Muon_dt_fIBeta     ,"Muon_dt_fIBeta[NMuons]/F");
   MyTree->Branch("Muon_dt_fIBeta_err" ,Muon_dt_fIBeta_err ,"Muon_dt_fIBeta_err[NMuons]/F");
   MyTree->Branch("Muon_csc_IBeta"     ,Muon_csc_IBeta     ,"Muon_csc_IBeta[NMuons]/F");
   MyTree->Branch("Muon_csc_IBeta_err" ,Muon_csc_IBeta_err ,"Muon_csc_IBeta_err[NMuons]/F");
   MyTree->Branch("Muon_csc_fIBeta"    ,Muon_csc_fIBeta    ,"Muon_csc_fIBeta[NMuons]/F");
   MyTree->Branch("Muon_csc_fIBeta_err",Muon_csc_fIBeta_err,"Muon_csc_fIBeta_err[NMuons]/F");
   MyTree->Branch("Muon_cb_IBeta"      ,Muon_cb_IBeta      ,"Muon_cb_IBeta[NMuons]/F");
   MyTree->Branch("Muon_cb_IBeta_err"  ,Muon_cb_IBeta_err  ,"Muon_cb_IBeta_err[NMuons]/F");
   MyTree->Branch("Muon_cb_fIBeta"     ,Muon_cb_fIBeta     ,"Muon_cb_fIBeta[NMuons]/F");
   MyTree->Branch("Muon_cb_fIBeta_err" ,Muon_cb_fIBeta_err ,"Muon_cb_fIBeta_err[NMuons]/F");
   MyTree->Branch("Muon_TrackIndex"    ,Muon_TrackIndex    ,"Muon_TrackIndex[NMuons]/i");
   MyTree->Branch("Muon_Track_pt"      ,Muon_Track_pt      ,"Muon_Track_pt[NMuons]/F");
   MyTree->Branch("Muon_Track_eta"     ,Muon_Track_eta     ,"Muon_Track_eta[NMuons]/F");
   MyTree->Branch("Muon_Track_phi"     ,Muon_Track_phi     ,"Muon_Track_phi[NMuons]/F");
   }


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

   // MUON LOOP:
   std::vector<reco::MuonRef> MuonVector;
   if(reccordMuonInfo){
      edm::Handle<reco::MuonCollection> muonCollectionHandle;
      iEvent.getByLabel(m_muonsTag,muonCollectionHandle);
      reco::MuonCollection muonCollection = *muonCollectionHandle.product();

      Handle<reco::MuonTimeExtraMap> timeMapDT_h;
      Handle<reco::MuonTimeExtraMap> timeMapCSC_h;
      Handle<reco::MuonTimeExtraMap> timeMapCmb_h;

      if(muonCollection.size()){
         iEvent.getByLabel(m_muontimingTag.label(),"dt",timeMapDT_h);
         iEvent.getByLabel(m_muontimingTag.label(),"csc",timeMapCSC_h);
         iEvent.getByLabel(m_muontimingTag.label(),"combined",timeMapCmb_h);
      }
      const reco::MuonTimeExtraMap & timeMapDT = *timeMapDT_h;
      const reco::MuonTimeExtraMap & timeMapCSC = *timeMapCSC_h;
      const reco::MuonTimeExtraMap & timeMapCmb = *timeMapCmb_h;



      NMuons = 0;
      for(unsigned int i=0; i<muonCollection.size(); i++){
        reco::MuonRef muon  = reco::MuonRef( muonCollectionHandle, i );
        MuonVector.push_back(muon);

        Muon_p             [NMuons] = muon->p();
        Muon_px            [NMuons] = muon->px();
        Muon_py            [NMuons] = muon->py();
        Muon_pz            [NMuons] = muon->pz();
        Muon_pt            [NMuons] = muon->pt();
      //Muon_chi2          [NMuons] = muon->chi2();
      //Muon_ndof          [NMuons] = muon->ndof();
        Muon_eta           [NMuons] = muon->eta();
        Muon_phi           [NMuons] = muon->phi();
        Muon_theta         [NMuons] = muon->theta();
        Muon_type          [NMuons] = muon->type();
      //Muon_quality       [NMuons] = muon->quality();
        Muon_qualityValid  [NMuons] = muon->isQualityValid();
        Muon_charge        [NMuons] = muon->charge();
        Muon_dt_IBeta      [NMuons] = timeMapDT [muon].inverseBeta();
        Muon_dt_IBeta_err  [NMuons] = timeMapDT [muon].inverseBetaErr();
        Muon_dt_fIBeta     [NMuons] = timeMapDT [muon].freeInverseBeta();
        Muon_dt_fIBeta_err [NMuons] = timeMapDT [muon].freeInverseBetaErr();
        Muon_csc_IBeta     [NMuons] = timeMapCSC[muon].inverseBeta();
        Muon_csc_IBeta_err [NMuons] = timeMapCSC[muon].inverseBetaErr();
        Muon_csc_fIBeta    [NMuons] = timeMapCSC[muon].freeInverseBeta();
        Muon_csc_fIBeta_err[NMuons] = timeMapCSC[muon].freeInverseBetaErr();
        Muon_cb_IBeta      [NMuons] = timeMapCmb[muon].inverseBeta();
        Muon_cb_IBeta_err  [NMuons] = timeMapCmb[muon].inverseBetaErr();
        Muon_cb_fIBeta     [NMuons] = timeMapCmb[muon].freeInverseBeta();
        Muon_cb_fIBeta_err [NMuons] = timeMapCmb[muon].freeInverseBetaErr();

        TrackRef innertrack = muon->innerTrack();
        if(innertrack.isNonnull()){ 
           Muon_TrackIndex    [NMuons] = innertrack.key();
           Muon_Track_pt      [NMuons] = innertrack->pt ();
           Muon_Track_eta     [NMuons] = innertrack->eta();
           Muon_Track_phi     [NMuons] = innertrack->phi();
        }else{
           Muon_TrackIndex    [NMuons] = -1;
           Muon_Track_pt      [NMuons] = -1;
           Muon_Track_eta     [NMuons] = -1;
           Muon_Track_phi     [NMuons] = -1;
        }

        NMuons++;
      }
   }


   if(reccordTrackInfo){
   // GET BEAMSPOT:
   edm::Handle<reco::BeamSpot> beamSpotHandle;
   iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
   reco::BeamSpot beamSpot = *beamSpotHandle;

   // TRACK LOOP:
   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   try { iEvent.getByLabel(m_tracksTag,trackCollectionHandle); } catch (...) {;}
   reco::TrackCollection trackCollection = *trackCollectionHandle.product();

   NTracks = 0;
   for(unsigned int i=0; i<trackCollection.size(); i++){
     reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );

     if(fabs(track->eta()) <MinTrackEta      ||  fabs(track->eta())>MaxTrackEta)       continue;
     if(track->p()         <MinTrackMomentum ||  track->p()        >MaxTrackMomentum)  continue;
     if(track->pt()        <MinTrackTMomentum||  track->pt()       >MaxTrackTMomentum) continue;
     if(track->found()     <MinTrackNHits    ||  track->found()    >MaxTrackNHits)     continue;

     Track_p        [NTracks] = track->p();
     Track_px       [NTracks] = track->px();
     Track_py       [NTracks] = track->py();
     Track_pz       [NTracks] = track->pz();
     Track_pt       [NTracks] = track->pt();
     Track_pt_err   [NTracks] = track->ptError();
     Track_eta      [NTracks] = track->eta();
     Track_eta_err  [NTracks] = track->etaError();
     Track_phi      [NTracks] = track->phi();
     Track_phi_err  [NTracks] = track->phiError();
     Track_theta    [NTracks] = track->theta();
     Track_NOH      [NTracks] = track->found();
     Track_chi2     [NTracks] = track->chi2();
     Track_ndof     [NTracks] = track->ndof();
//   Track_d0       [NTracks] = -1.0f * track->dxy(math::XYZPoint(beamSpot.x0(),beamSpot.y0(), beamSpot.z0()));
//   Track_dz       [NTracks] = -1.0f * track->dz (math::XYZPoint(beamSpot.x0(),beamSpot.y0(), beamSpot.z0()));
     Track_d0       [NTracks] = -1.0f * track->dxy(recoVertex[0].position());
     Track_dz       [NTracks] = -1.0f * track->dz (recoVertex[0].position());
     Track_quality  [NTracks] = track->qualityMask();
     Track_charge   [NTracks] = track->charge(); 
     Track_MuonIndex[NTracks] = ClosestMuonIndex(track, MuonVector);
     if(Track_MuonIndex[NTracks]>=0){
     reco::MuonRef muon = MuonVector[ Track_MuonIndex[NTracks] ];     
     Track_MuonDist [NTracks] = deltaR(track->eta(), track->phi(),muon->eta(), muon->phi());
     }else{
     Track_MuonDist [NTracks] = -1;
     }

     for(unsigned int j=0;j<m_dEdxDiscrimTag.size();j++){
        Handle<ValueMap<DeDxData> > dEdxTrackHandle;
        try { iEvent.getByLabel(m_dEdxDiscrimTag[j], dEdxTrackHandle); } catch (...) {;}
        const ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

        (Track_dEdx    [j])[NTracks] = dEdxTrack[track].dEdx();
        (Track_dEdx_NOS[j])[NTracks] = dEdxTrack[track].numberOfSaturatedMeasurements();
        (Track_dEdx_NOM[j])[NTracks] = dEdxTrack[track].numberOfMeasurements();       
     }
  
     NTracks++; 
   }
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



int HSCPTreeBuilder::ClosestMuonIndex(reco::TrackRef track, std::vector<reco::MuonRef> muons){
   double RMin      = 1000;
   int    MuonIndex = -1;
   for (unsigned int i=0; i<muons.size(); i++){
      reco::MuonRef muon = muons[i];
      if(!muon->isQualityValid())continue;
      double dr = deltaR(track->eta(), track->phi(),muon->eta(), muon->phi());
      if(dr<RMin){
         MuonIndex = i;
         RMin      = dr;
      }
   }
   return MuonIndex;
}


//define this as a plug-in
DEFINE_FWK_MODULE(HSCPTreeBuilder);










