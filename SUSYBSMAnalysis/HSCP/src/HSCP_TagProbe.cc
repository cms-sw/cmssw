/** \class HSCP_TagProbe
 *  Example selector of muons.
 *
 *  $Date: 2010/10/12 16:46:13 $
 *  $Revision: 1.2 $
 *  \author G. Petrucciani (SNS Pisa)
 */

// Base Class Headers
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

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"

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
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

using namespace edm;
using namespace std;

class HSCP_TagProbe : public edm::EDProducer {

public:
  /// constructor
  HSCP_TagProbe(const edm::ParameterSet &iConfig) ;
  /// destructor
  ~HSCP_TagProbe() ;
    
  /// method to be called at each event
  virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;
  double DistTrigger(const reco::TrackRef muonTrack, const edm::Event& iEvent);
  double SegSep(const reco::TrackRef muonTrack, const edm::Event& iEvent, const edm::EventSetup &iSetup);
  int  muonStations(reco::HitPattern hitPattern);

private:
  //--- put here your data members ---

  /// Input collection of muons
  edm::InputTag src_;
  edm::InputTag SAtracks_;
  edm::InputTag Tktracks_;
  edm::InputTag input_dedx_collection;

  bool SACut;
  bool TkCut;
  bool GlbCut;
  double DzCut;
  double DxyCut;
  int StationCut;
  int TOFNdofCut;
  double TOFErrCut;
  double SegSepCut;
  double SAPtCut;
  double SAEtaCut;
  int minTrackHits;
  int minPixelHits;
  unsigned int minDeDxMeas;
  double maxV3D;
  int minQualityMask;
  double maxTrackIso;
  double maxEoP;
  double minFraction;
  double maxChi2;
  double maxPtErr;
  double minPhi;
  double maxPhi;
  double timeRange;
}; // C++ note: you need a ';' at the end of the class declaration.


/// Constructor: read the configuration, initialize data members, declare what to produce
HSCP_TagProbe::HSCP_TagProbe(const edm::ParameterSet &iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  SAtracks_(iConfig.getParameter<edm::InputTag>("SAtracks")),
  Tktracks_(iConfig.getParameter<edm::InputTag>("Tktracks")),
  input_dedx_collection(iConfig.getParameter< edm::InputTag >("inputDedxCollection")),
  SACut(iConfig.getParameter<bool>("SACut")),
  TkCut(iConfig.getParameter<bool>("TkCut")),
  DzCut(iConfig.getParameter<double>("DzCut")),
  DxyCut(iConfig.getParameter<double>("DxyCut")),
  StationCut(iConfig.getParameter<int>("StationCut")),
  TOFNdofCut(iConfig.getParameter<int>("TOFNdofCut")),
  TOFErrCut(iConfig.getParameter<double>("TOFErrCut")),
  SegSepCut(iConfig.getParameter<double>("SegSepCut")),
  SAPtCut(iConfig.getParameter<double>("SAPtCut")),
  SAEtaCut(iConfig.getParameter<double>("SAEtaCut")),
  minTrackHits(iConfig.getParameter<int>("minTrackHits")),
  minPixelHits(iConfig.getParameter<int>("minPixelHits")),
  minDeDxMeas(iConfig.getParameter<uint>("minDeDxMeas")),
  maxV3D(iConfig.getParameter<double>("maxV3D")),
  minQualityMask(iConfig.getParameter<int>("minQualityMask")),
  maxTrackIso(iConfig.getParameter<double>("maxTrackIso")),
  maxEoP(iConfig.getParameter<double>("maxEoP")),
  minFraction(iConfig.getParameter<double>("minFraction")),
  maxChi2(iConfig.getParameter<double>("maxChi2")),
  maxPtErr(iConfig.getParameter<double>("maxPtErr")),
  minPhi(iConfig.getParameter<double>("minPhi")),
  maxPhi(iConfig.getParameter<double>("maxPhi")),
  timeRange(iConfig.getParameter<double>("timeRange"))
{
  // declare what we produce: a vector of references to Muon-like objects (can be reco::Muon or pat::Muons)
  // subsequent modules should read this with View<reco::Muon>
  produces<edm::RefToBaseVector<reco::Muon> >();    
}

/// Destructor: here there is usually nothing to do
HSCP_TagProbe::~HSCP_TagProbe() 
{
}

/// Produce: the method where we do something
void 
HSCP_TagProbe::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) 
{
  using namespace edm;
  using namespace std;
  using namespace reco;


  // read the input from the event.
  // View<reco::Muon> can read any collection of reco::Muon or pat::Muon, or any collection of references to them
  Handle<View<reco::Muon> > src;
  iEvent.getByLabel(src_, src);

  // If you need to read other stuff, e.g. other collections, associations or condition data, do it here.

  Handle<View<reco::Track> > SAtracks;
  iEvent.getByLabel(SAtracks_, SAtracks);


  edm::Handle<reco::TrackCollection> Tktracks;
  iEvent.getByLabel(Tktracks_, Tktracks);

  edm::Handle<reco::BeamSpot> beamSpotCollHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotCollHandle);
  if(!beamSpotCollHandle.isValid()){printf("BeamSpot Collection NotFound\n");}
  const reco::BeamSpot& beamSpotColl = *beamSpotCollHandle;

  edm::Handle<reco::VertexCollection> Vertex;
  iEvent.getByLabel("offlinePrimaryVertices", Vertex);

  edm::Handle<reco::MuonTimeExtraMap> timeMap;
  edm::Handle<reco::MuonTimeExtraMap> timeDTMap;
  edm::Handle<reco::MuonTimeExtraMap> timeCSCMap;

  iEvent.getByLabel("muontiming","combined", timeMap);
  const reco::MuonTimeExtraMap & timeMapCmb = *timeMap;

  //iEvent.getByLabel("muontiming","dt", timeDTMap);
  //const reco::MuonTimeExtraMap & timeMapDT = *timeDTMap;

  //iEvent.getByLabel("muontiming","csc", timeCSCMap);
  //const reco::MuonTimeExtraMap & timeMapCSC = *timeCSCMap;

  edm::Handle<edm::ValueMap<DeDxData> >     dEdxTrackHandle;
  iEvent.getByLabel(input_dedx_collection, dEdxTrackHandle);
  const edm::ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

  Handle<susybsm::HSCPIsolationValueMap> IsolationH;
  iEvent.getByLabel("HSCPIsolation03", IsolationH);
  if(!IsolationH.isValid()){printf("Invalid IsolationH\n");return;}
  const ValueMap<susybsm::HSCPIsolation>& IsolationMap = *IsolationH.product();

  /// prepare the vector for the output
  std::auto_ptr<RefToBaseVector<reco::Muon> > out(new RefToBaseVector<reco::Muon>());

    ///// Now loop
    for (size_t i = 0, n = src->size(); i < n; ++i) {
    // read the edm reference to the muon
    RefToBase<reco::Muon> srcRef = src->refAt(i);
    const reco::Muon & src = *srcRef;

    reco::MuonTimeExtra timec   = timeMapCmb[srcRef];
    //reco::MuonTimeExtra timedt  = timeMapDT[srcRef];
    //reco::MuonTimeExtra timecsc = timeMapCSC[srcRef];

    const reco::TrackRef SATrackRef = src.standAloneMuon();
    if(SACut && SATrackRef.isNull()) continue;

    if(!SATrackRef.isNull() && SACut) {
      if(SATrackRef->pt()<SAPtCut) continue;
      if(fabs(SATrackRef->eta())>SAEtaCut) continue;
      if(muonStations(SATrackRef->hitPattern())<StationCut) continue;
    double dr2min = 10000;
    int match=-1;
    for (unsigned int iTk=0; iTk<SAtracks->size(); iTk++) {
      const reco::Track Tk = SAtracks->at(iTk);
      double dr2 = deltaR2(src, Tk);
      if (dr2 < dr2min && dr2<0.3) { match=iTk; dr2min=dr2;}
    }
    double dz=999;
    double dxy=999;
    if(match>-1) {
      const reco::Track Tk = SAtracks->at(match);
      dz=fabs(Tk.dz(beamSpotColl.position()));
      dxy=fabs(Tk.dxy(beamSpotColl.position()));
      //if(fabs(dz)<6) out->push_back(muonRef);
    }

    if(dz>DzCut) continue;

    if(dxy>DxyCut) continue;

    if(fabs(SegSep(SATrackRef, iEvent, iSetup))<SegSepCut) continue;
    }

    if(timec.nDof()<TOFNdofCut) continue;

    if(timec.inverseBetaErr()>TOFErrCut) continue;

    if(maxPhi>0 && fabs(SATrackRef->phi())>minPhi && fabs(SATrackRef->phi()<maxPhi)) continue;

    if(timeRange>0)
      if(min(min(fabs(timec.timeAtIpInOut()-100), fabs(timec.timeAtIpInOut()-50)), min(fabs(timec.timeAtIpInOut()+100), fabs(timec.timeAtIpInOut()+50)))<timeRange) continue;

    if(TkCut) {
      double dr2min = 10000;
      int match=-1;

      for (unsigned int iTk=0; iTk<Tktracks->size(); iTk++) {
	TrackRef track = reco::TrackRef(Tktracks, iTk );
	//const reco::Track Tk = SAtracks->at(iTk);
	//if( fabs( (1.0/innertrack->pt())-(1.0/track->pt())) > 0.005) continue;
	float dR = deltaR(src.momentum(), track->momentum());
	if (dR < dr2min && dR<0.3) { match=iTk; dr2min=dR;}
      }

      if(match>-1) {
        TrackRef track = reco::TrackRef(Tktracks, match);

	if(track->found()<minTrackHits)continue;

        if(track->hitPattern().pixelLayersWithMeasurement()<minPixelHits)continue;
	//if(track->hitPattern().numberOfValidPixelHits()<minPixelHits)continue;
	if(track->validFraction()<minFraction)continue;

	if(dEdxTrack[track].numberOfMeasurements()<minDeDxMeas)continue;

	if(track->qualityMask()<minQualityMask)continue;
	if(track->chi2()/track->ndof()>maxChi2)continue;
	//if(dEdxTrack[track].dEdx()>3.0)continue;

	double dz  = track->dz (Vertex->begin()->position());
	double dxy = track->dxy(Vertex->begin()->position());
	for (unsigned int iVert=0; iVert<Vertex->size(); iVert++) {
	  const reco::Vertex iVertex = Vertex->at(iVert);
	  if(fabs(track->dz (iVertex.position())) < fabs(dz) ){
	    dz  = track->dz (iVertex.position());
	    dxy = track->dxy(iVertex.position());
	  }
	}
	double v3d = sqrt(dz*dz+dxy*dxy);

	if(v3d>maxV3D)continue;

	susybsm::HSCPIsolation hscpIso = IsolationMap.get((size_t)track.key());
	if(hscpIso.Get_TK_SumEt()>maxTrackIso)continue;
	double EoP = (hscpIso.Get_ECAL_Energy() + hscpIso.Get_HCAL_Energy())/track->p();
	if(EoP>maxEoP)continue;

	if((track->ptError()/track->pt())>maxPtErr)continue;
      }
    }
    out->push_back(srcRef);
  }
  // Write the output to the event
  iEvent.put(out);
}

//Counts the number of muon stations used in track fit only counting DT and CSC stations.
int  HSCP_TagProbe::muonStations(reco::HitPattern hitPattern) {
  int stations[4] = { 0,0,0,0 };

  for (int i=0; i<hitPattern.numberOfHits(); i++) {
    uint32_t pattern = hitPattern.getHitPattern(i);
    if (pattern == 0) break;
    if (hitPattern.muonHitFilter(pattern) &&
        (int(hitPattern.getSubStructure(pattern)) == 1 ||
         int(hitPattern.getSubStructure(pattern)) == 2) &&
        hitPattern.getHitType(pattern) == 0) {
      stations[hitPattern.getMuonStation(pattern)-1] = 1;
    }
  }

  return stations[0]+stations[1]+stations[2]+stations[3];
}

double HSCP_TagProbe::SegSep(const reco::TrackRef muonTrack, const edm::Event& iEvent, const edm::EventSetup &iSetup) {

  /*
  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  edm::Handle<DTRecSegment4DCollection> dtSegments;
  iEvent.getByLabel("dt4DSegmentsMT", dtSegments);

  double minEta=10;

  double eta_track = -1*muonTrack->eta();
  double phi_track= muonTrack->phi()+M_PI;

  for (unsigned int d=0; d<dtSegments->size(); d++) {
    DTRecSegment4DRef SegRef  = DTRecSegment4DRef( dtSegments, d );
    const GeomDet* dtDet = dtGeom->idToDet(SegRef->geographicalId());
    GlobalPoint point = dtDet->toGlobal(SegRef->localPosition());

    double deta = point.eta() - eta_track;
    double dphi = point.phi() - phi_track;
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;

    //Find segment most opposite in eta  
    //Require phi difference of 0.5 so it doesn't match to own segment
    if(fabs(deta)<fabs(minEta) && fabs(dphi)<(M_PI-0.5)) {
      minEta=deta;
    }
  }

  edm::Handle<CSCSegmentCollection> cscSegments;
  iEvent.getByLabel("cscSegments", cscSegments);

  for (unsigned int c=0; c<cscSegments->size(); c++) {
    CSCSegmentRef SegRef  = CSCSegmentRef( cscSegments, c );
    const GeomDet* cscDet = cscGeom->idToDet(SegRef->geographicalId());
    GlobalPoint point = cscDet->toGlobal(SegRef->localPosition());

    double deta = point.eta() - eta_track;
    double dphi = point.phi() - phi_track;
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;

    //Find segment most opposite in eta  
    //Require phi difference of 0.5 so it doesn't match to own segment
    if(fabs(deta)<fabs(minEta) && fabs(dphi)<(M_PI-0.5)) {
      minEta=deta;
    }
  }
  return minEta;
  */

  /*
  fwlite::Handle<MuonSegmentCollection> SegCollHandle;
  SegCollHandle.getByLabel(ev, "MuonSegmentProducer");
  if(!SegCollHandle.isValid()){printf("Segment Collection Not Found\n"); return -1;}
  MuonSegmentCollection SegCollection = *SegCollHandle;
  */
  edm::Handle<susybsm::MuonSegmentCollection> SegCollHandle;
  iEvent.getByLabel("MuonSegmentProducer", SegCollHandle);

  double minEta=10;

  //Look for segment on opposite side of detector from track
  for (unsigned int c=0; c<SegCollHandle->size(); c++) {
    susybsm::MuonSegmentRef SegRef  = susybsm::MuonSegmentRef( SegCollHandle, c );

    GlobalPoint gp = SegRef->getGP();

    //Flip HSCP to opposite side of detector
    double eta_hscp = -1*muonTrack->eta();
    double phi_hscp= muonTrack->phi()+M_PI;

    double deta = gp.eta() - eta_hscp;
    double dphi = gp.phi() - phi_hscp;
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;

    //Find segment most opposite in eta
    //Require phi difference of 0.5 so it doesn't match to own segment
    if(fabs(deta)<fabs(minEta) && fabs(dphi)<(M_PI-0.5)) {
      minEta=deta;
    }
  }
  return minEta;

}

/// Register this as a CMSSW Plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HSCP_TagProbe);
