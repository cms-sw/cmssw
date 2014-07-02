///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of L1TkJetParticle,                                          //
// associating L1 jets to a z vertex using nearby L1 tracks              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"

#include "DataFormats/Math/interface/LorentzVector.h"

// L1 tracks
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


// geometry
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"


#include <string>
#include "TMath.h"
#include "TH1.h"

using namespace l1extra;
using namespace edm;
using namespace std;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TkJetProducer : public edm::EDProducer 
{
public:
  
   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >    L1TkTrackCollectionType;
  
  explicit L1TkJetProducer(const edm::ParameterSet&);
  ~L1TkJetProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  //virtual void beginRun(edm::Run&, edm::EventSetup const&);
  //virtual void endRun(edm::Run&, edm::EventSetup const&);
  //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  
  // member data
  edm::InputTag L1CentralJetInputTag;
  edm::InputTag L1TrackInputTag;

  // track selection criteria
  float TRK_ZMAX;       // [cm]
  float TRK_CHI2MAX;    // maximum track chi2
  float TRK_PTMIN;      // [GeV]
  float TRK_ETAMAX;     // [rad]
  int   TRK_NSTUBMIN;   // minimum number of stubs 
  int   TRK_NSTUBPSMIN; // minimum number of stubs in PS modules 

  // jet cut 
  //bool JET_HLTETA;  // temporary hack to remove bad jets when using HI HLT jets!

  bool doPtComp;
  bool doTightChi2;

  // geometry for stub info
  const StackedTrackerGeometry* theStackedGeometry;

};

//////////////
// constructor
L1TkJetProducer::L1TkJetProducer(const edm::ParameterSet& iConfig)
{
  
  L1CentralJetInputTag = iConfig.getParameter<edm::InputTag>("L1CentralJetInputTag");
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  
  produces<L1TkJetParticleCollection>("Central");
  
  TRK_ZMAX    = (float)iConfig.getParameter<double>("TRK_ZMAX");
  TRK_CHI2MAX = (float)iConfig.getParameter<double>("TRK_CHI2MAX");
  TRK_PTMIN   = (float)iConfig.getParameter<double>("TRK_PTMIN");
  TRK_ETAMAX  = (float)iConfig.getParameter<double>("TRK_ETAMAX");
  TRK_NSTUBMIN   = (int)iConfig.getParameter<int>("TRK_NSTUBMIN");
  TRK_NSTUBPSMIN = (int)iConfig.getParameter<int>("TRK_NSTUBPSMIN");
  //JET_HLTETA = (bool)iConfig.getParameter<bool>("JET_HLTETA");
  doPtComp     = iConfig.getParameter<bool>("doPtComp");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");
  
}

/////////////
// destructor
L1TkJetProducer::~L1TkJetProducer() {
}

///////////
// producer
void L1TkJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  
  // ----------------------------------------------------------------------------------------------
  // output container
  // ----------------------------------------------------------------------------------------------

  std::auto_ptr<L1TkJetParticleCollection> cenTkJets(new L1TkJetParticleCollection);

  
  // ----------------------------------------------------------------------------------------------
  // retrieve input containers 
  // ----------------------------------------------------------------------------------------------

    // L1 jets
  edm::Handle<L1JetParticleCollection> CentralJetHandle;
  iEvent.getByLabel(L1CentralJetInputTag,CentralJetHandle);
  std::vector<L1JetParticle>::const_iterator jetIter;

  // L1 tracks
  edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
  iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
  L1TkTrackCollectionType::const_iterator trackIter;

  // geometry handles (for stub info)
  edm::ESHandle<StackedTrackerGeometry> StackedGeometryHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product();


  // ----------------------------------------------------------------------------------------------
  // central jets (i.e. |eta| < 3)
  // ----------------------------------------------------------------------------------------------
  
  if ( !CentralJetHandle.isValid() ) {
    LogError("L1TkJetProducer")
      << "\nWarning: L1JetParticleCollection with " << L1CentralJetInputTag
      << "\nrequested in configuration, but not found in the event."
      << std::endl;
  }
  else {
    
    // ----------------------------------------------------------------------------------------------
    // loop over jets
    // ----------------------------------------------------------------------------------------------

    // make these configurable ??
    float max_deltaR = 0.4;
    float max_z0_outlier1 = 5; //cm
    float max_z0_outlier2 = 2; //cm
    //cout << "Using outlier removal (1) |dz| < " << max_z0_outlier1 << " cm, and (2) |dz| < " << max_z0_outlier2 << " cm..." << endl; 

    float trk_sum_z0 = 0;
    float trk_sum_pt = 0;
    float jet_z0_v0 = 999;
    float jet_z0_v1 = 999;
    float jet_z0_v2 = 999;
    
    float sumTrk_pt = 0;

    int ijet = 0;
    for (jetIter = CentralJetHandle->begin();  jetIter != CentralJetHandle->end(); ++jetIter) {

      edm::Ref< L1JetParticleCollection > jetRef(CentralJetHandle, ijet);
      ijet++;

      // kinematics
      //float tmp_jet_pt  = jetIter->pt();
      float tmp_jet_eta = jetIter->eta();
      float tmp_jet_phi = jetIter->phi();

      // for HI-HLT jets, remove jets from "hot spots"
      //if (JET_HLTETA) {
	//if ( (tmp_jet_eta > -2.0) && (tmp_jet_eta < -1.5) && (tmp_jet_phi > -1.7) && (tmp_jet_phi < -1.3) ) continue;
	//if ( (tmp_jet_eta > 0.0) && (tmp_jet_eta < 1.4) && (tmp_jet_phi > -2.0) && (tmp_jet_phi < -1.6) ) continue;
      //}

      // only consider jets from the central BX
      int ibx = jetIter->bx();
      if (ibx != 0) continue;


      // ----------------------------------------------------------------------------------------------
      // calculate the vertex of the jet
      // ----------------------------------------------------------------------------------------------

      float this_zpos = 999.;
      bool this_continue = true;
      
      TH1F* h_start_trk = new TH1F("start_trk", "; z_{0} [cm]; # tracks / 2cm", 25,-25,25);

      sumTrk_pt = 0;
      
      std::vector<double> v_trk_pt;
      std::vector<double> v_trk_z0;

      
      // ----------------------------------------------------------------------------------------------------------------
      // track loop
      // ----------------------------------------------------------------------------------------------------------------

      std::vector< edm::Ptr< L1TkTrackType > > L1TrackPtrs;
      int itrk = -1;

      for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {
	
	itrk++;

	float tmp_trk_pt   = trackIter->getMomentum().perp();
	float tmp_trk_eta  = trackIter->getMomentum().eta();
	float tmp_trk_phi  = trackIter->getMomentum().phi();
	float tmp_trk_z0   = trackIter->getPOCA().z();
	float tmp_trk_chi2 = trackIter->getChi2();

	// get pointers to stubs associated to the L1 track
	//std::vector< edm::Ptr< L1TkStub_PixelDigi_ > > theStubs = trackIter->getStubPtrs();
	std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

	int tmp_trk_nstub = (int) theStubs.size();
	//int tmp_ndof = tmp_trk_nstub*2-4;
	
	// track selection
	if (fabs(tmp_trk_eta) > TRK_ETAMAX) continue;
	if (tmp_trk_pt < TRK_PTMIN) continue;
	if (fabs(tmp_trk_z0) > TRK_ZMAX) continue;
	if (tmp_trk_chi2 > TRK_CHI2MAX) continue;
	if (tmp_trk_nstub < TRK_NSTUBMIN) continue;

	// loop over the stubs
	int tmp_trk_nstubPS = 0;
	for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
	  StackedTrackerDetId detIdStub(theStubs.at(istub)->getDetId());
	  bool tmp_isPS = theStackedGeometry->isPSModule(detIdStub);
	  if (tmp_isPS) tmp_trk_nstubPS++;
	}

	// more track selection
	if (tmp_trk_nstubPS < TRK_NSTUBPSMIN) continue;


	////_______                                                                                                                                                         
	////-------                                                                                                                                                         
	float tmp_trk_consistency = trackIter ->getStubPtConsistency();
	float chi2dof = tmp_trk_chi2 / (2*tmp_trk_nstub-4);
	
	if(doPtComp) {
	  //              if (trk_nstub < 4) continue;                                                                                                                      
	  //              if (trk_chi2 > 100.0) continue;                                                                                                                   
	  if (tmp_trk_nstub == 4) {
	    if (fabs(tmp_trk_eta)<2.2 && tmp_trk_consistency>10) continue;
	    else if (fabs(tmp_trk_eta)>2.2 && chi2dof>5.0) continue;
	  }
	}

            if(doTightChi2) {
              if(tmp_trk_pt>10.0 && chi2dof>5.0 ) continue;     
            }      
	
	
	////_______                                                                                                                                                         
	////-------                                                                                                                                                         
	



	// deltaR
	float deltaEta = fabs(tmp_jet_eta-tmp_trk_eta);
	float deltaPhi = fabs(tmp_jet_phi-tmp_trk_phi);
	while (deltaPhi > 3.14159) deltaPhi = fabs(2*3.14159 - deltaPhi);
	float deltaR = sqrt( deltaEta*deltaEta + deltaPhi*deltaPhi );

	if (deltaR < max_deltaR) { //use tracks within deltaR < max_deltaR of the jet
	  v_trk_z0.push_back(tmp_trk_z0);
	  v_trk_pt.push_back(tmp_trk_pt);
	  sumTrk_pt += tmp_trk_pt;
	  
	  edm::Ptr< L1TkTrackType > trkPtr( L1TkTrackHandle, itrk) ;
	  L1TrackPtrs.push_back(trkPtr);
	  
	  h_start_trk->Fill(tmp_trk_z0);
	}

      }//end track loop
      // ----------------------------------------------------------------------------------------------------------------

      jet_z0_v0 = 999;
      jet_z0_v1 = 999;
      jet_z0_v2 = 999;
      
      int ntrkCone = v_trk_z0.size();

      // ----------------------------------------------------------------------------------------------------------------
      // STEP (0)
      // ----------------------------------------------------------------------------------------------------------------
      if (ntrkCone < 1) {
	this_zpos = 999.;
	this_continue = false; 
	h_start_trk->Delete();
      }

      if (this_continue) {
	
	// ----------------------------------------------------------------------------------------------------------------
	// STEP (1)
	// ----------------------------------------------------------------------------------------------------------------
	trk_sum_z0 = 0;
	trk_sum_pt = 0;
	for (int k=0; k<ntrkCone; k++) {
	  trk_sum_z0 += v_trk_z0.at(k)*v_trk_pt.at(k);
	  trk_sum_pt += v_trk_pt.at(k);
	}
     
	// ----------------------------------------------------------------------------------------------------------------
	// STEP (2)
	// ----------------------------------------------------------------------------------------------------------------
	//jet_z0_v0 = trk_sum_z0/trk_sum_pt;
	jet_z0_v0 = (float)h_start_trk->GetBinCenter(h_start_trk->GetMaximumBin());
	h_start_trk->Delete();


	// ----------------------------------------------------------------------------------------------------------------
	// STEP (3)
	// ----------------------------------------------------------------------------------------------------------------
	for (int k=0; k<ntrkCone; k++) {
	  if (fabs(v_trk_z0.at(k) - jet_z0_v0) > max_z0_outlier1) {
	    trk_sum_z0 -= v_trk_z0.at(k)*v_trk_pt.at(k);
	    trk_sum_pt -= v_trk_pt.at(k);
	    v_trk_z0.erase(v_trk_z0.begin()+k);
	    v_trk_pt.erase(v_trk_pt.begin()+k);
	    k--;
	    ntrkCone--;
	  }		
	}
	
	if (ntrkCone < 1) {
	  this_zpos = 999.;
	  this_continue = false; 
	}
      }

      if (this_continue) {
	
	// ----------------------------------------------------------------------------------------------------------------
	// STEP (4)
	// ----------------------------------------------------------------------------------------------------------------
	jet_z0_v1 = trk_sum_z0/trk_sum_pt;
	
	// ----------------------------------------------------------------------------------------------------------------
	// STEP (5)
	// ----------------------------------------------------------------------------------------------------------------
	for (int k=0; k<ntrkCone; k++) {
	  if (fabs(v_trk_z0.at(k) - jet_z0_v1) > max_z0_outlier2) {
	    trk_sum_z0 -= v_trk_z0.at(k)*v_trk_pt.at(k);
	    trk_sum_pt -= v_trk_pt.at(k);
	    v_trk_z0.erase(v_trk_z0.begin()+k);
	    v_trk_pt.erase(v_trk_pt.begin()+k);
	    k--;
	    ntrkCone--;
	  }	
	}
	
	if (ntrkCone < 1) {
	  this_zpos = 999.;
	  this_continue = false;
	}
      }

      if (this_continue) {

	// ----------------------------------------------------------------------------------------------------------------
	// STEP (6)
	// ----------------------------------------------------------------------------------------------------------------
	jet_z0_v2 = trk_sum_z0/trk_sum_pt;
	this_zpos = jet_z0_v2;

      }



      // ----------------------------------------------------------------------------------------------
      // end of jet loop, create the L1TkJetParticle and push to collection
      // ----------------------------------------------------------------------------------------------

      const math::XYZTLorentzVector jetP4 = jetIter->p4();
      L1TkJetParticle trkJet(jetP4, jetRef, L1TrackPtrs, this_zpos);
      
      cenTkJets->push_back(trkJet);


    } //end loop over jets
  } //endif CentralJetHandle.isValid()



 iEvent.put(cenTkJets, "Central");



}


// ------------ method called once each job just before starting event loop  ------------
void
L1TkJetProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkJetProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkJetProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkJetProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkJetProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkJetProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkJetProducer);



