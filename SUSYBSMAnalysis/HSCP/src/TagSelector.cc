#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"




#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/MuonSegment.h"





//
// class declaration
//
class TagSelector : public edm::EDFilter {
   public:
      explicit TagSelector(const edm::ParameterSet&);
      ~TagSelector();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      double TriggerDr(edm::RefToBase<reco::Muon> mu, edm::Event& Event);

      edm::InputTag sourceTag_;
      edm::InputTag muonTag_;
      edm::InputTag input_dedx_collection;

      bool isProbe;
};


/////////////////////////////////////////////////////////////////////////////////////
TagSelector::TagSelector(const edm::ParameterSet& pset)
{
   // What is being produced
   produces<edm::RefToBaseVector<reco::Muon> >();
    // Input products
   sourceTag_     = pset.getUntrackedParameter<edm::InputTag> ("source"    , edm::InputTag(""));
   muonTag_     = pset.getUntrackedParameter<edm::InputTag> ("muons"    , edm::InputTag(""));
   input_dedx_collection =  pset.getParameter< edm::InputTag >("inputDedxCollection");
   isProbe     = pset.getUntrackedParameter<bool> ("isProbe");
} 

/////////////////////////////////////////////////////////////////////////////////////
TagSelector::~TagSelector(){
}

/////////////////////////////////////////////////////////////////////////////////////
void TagSelector::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void TagSelector::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
bool TagSelector::filter(edm::Event& ev, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;
  using namespace reco;

      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollectionHandle;
      if (!ev.getByLabel(muonTag_, muonCollectionHandle)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return false;
      }

      // Source Collection
      edm::Handle<susybsm::HSCParticleCollection > SourceHandle;
      if (!ev.getByLabel(sourceTag_, SourceHandle)) {
            edm::LogError("") << ">>> HSCParticleCollection does not exist !!!";
            return false;
      }
      susybsm::HSCParticleCollection Source = *SourceHandle.product();

      unsigned int muonCollectionSize = muonCollectionHandle->size();
      /// prepare the vector for the output
      std::auto_ptr<edm::RefToBaseVector<reco::Muon> > out(new edm::RefToBaseVector<reco::Muon>());

      edm::Handle<reco::VertexCollection> Vertex;
      ev.getByLabel("offlinePrimaryVertices", Vertex);

      edm::Handle<edm::ValueMap<DeDxData> >     dEdxTrackHandle;
      ev.getByLabel(input_dedx_collection, dEdxTrackHandle);
      const edm::ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

      // cleanup the collection based on the input selection
      edm::RefToBase<reco::Muon> mu;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
	mu = muonCollectionHandle->refAt(i);
	bool matchFound=false;

      for(susybsm::HSCParticleCollection::iterator hscpcandidate = Source.begin(); hscpcandidate < Source.end(); ++hscpcandidate){
	if(!hscpcandidate->hasTrackRef() || !hscpcandidate->hasMuonRef()) continue;
	reco::MuonRef muon = hscpcandidate->muonRef();

	//edm::RefToBase<reco::Muon> mu;
        //for (unsigned int i=0; i<muonCollectionSize; i++) {
	//mu = muonCollectionHandle->refAt(i);
          if(fabs(hscpcandidate->muonRef()->pt()-mu->pt()) > 0.1) continue;
	  //}

	  if((!mu->isPFMuon() || !mu->isGlobalMuon()) && !isProbe ) continue;
	if(!muon::isGoodMuon(*mu,muon::GlobalMuonPromptTight) && !isProbe) continue;
	if(mu->numberOfMatchedStations() <2 && !isProbe) continue;

	if(hscpcandidate->trackRef()->hitPattern().trackerLayersWithMeasurement() < 6) continue;
	if(hscpcandidate->trackRef()->hitPattern().numberOfValidPixelHits() < 1) continue;
	if(fabs(hscpcandidate->trackRef()->dxy(Vertex->begin()->position())) > 0.2)continue;
	if(fabs(hscpcandidate->trackRef()->dz(Vertex->begin()->position())) > 0.5) continue;

	double dedx = dEdxTrack[hscpcandidate->trackRef()].dEdx();
	if(dedx <3.0 && dedx>2.8 && !isProbe) continue;
	int dedxnhits  = dEdxTrack[hscpcandidate->trackRef()].numberOfMeasurements();
	if(dedxnhits<3 && !isProbe)continue;
	matchFound=true;

      }
      if(matchFound) {
        out->push_back(mu);
	continue;
      }
      }
      if(out->size()==0) return false;
      ev.put(out);
      return true;
}

double TagSelector::TriggerDr(edm::RefToBase<reco::Muon> mu,edm::Event& Event) {
  edm::Handle< trigger::TriggerEvent > trEvHandle;
  Event.getByLabel("hltTriggerSummaryAOD", trEvHandle);
  trigger::TriggerEvent trEv = *trEvHandle;

  double minDr=9999;

  unsigned int filterIndex = trEv.filterIndex(edm::InputTag("hltL3fL1sMu16Eta2p1L1f0L2f16QL3Filtered40Q","","HLT"));
  if (filterIndex<trEv.sizeFilters()){
    const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
    const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
    const int nI(VIDS.size());
    const int nK(KEYS.size());
    assert(nI==nK);
    const int n(std::max(nI,nK));
    const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

    for (int i=0; i!=n; ++i) {
      double deta = mu->eta() - TOC[KEYS[i]].eta();
      double dphi = mu->phi() - TOC[KEYS[i]].phi();
      while (dphi >   M_PI) dphi -= 2*M_PI;
      while (dphi <= -M_PI) dphi += 2*M_PI;
      double dR = sqrt(deta*deta + dphi*dphi);
      if(dR<minDr) minDr=dR;
    }
  }
  else {
    std::cout << "Can't find trigger object" << std::endl;
  }
  return minDr;
}

DEFINE_FWK_MODULE(TagSelector);
