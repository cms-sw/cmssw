//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra: UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#include "SimMuon/MCTruth/plugins/MuonTrackProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include <sstream>

MuonTrackProducer::MuonTrackProducer(const edm::ParameterSet& parset) :
  muonsToken(consumes<reco::MuonCollection>(parset.getParameter< edm::InputTag >("muonsTag"))),
  inputDTRecSegment4DToken_(consumes<DTRecSegment4DCollection>(parset.getParameter<edm::InputTag>("inputDTRecSegment4DCollection"))),
  inputCSCSegmentToken_(consumes<CSCSegmentCollection>(parset.getParameter<edm::InputTag>("inputCSCSegmentCollection"))),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  trackType(parset.getParameter< std::string >("trackType")),
  parset_(parset)
{
  edm::LogVerbatim("MuonTrackProducer") << "constructing  MuonTrackProducer" << parset_.dump();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
}

MuonTrackProducer::~MuonTrackProducer() {
}

void MuonTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  iEvent.getByToken(muonsToken,muonCollectionH);
  iEvent.getByToken(inputDTRecSegment4DToken_, dtSegmentCollectionH_);
  iEvent.getByToken(inputCSCSegmentToken_, cscSegmentCollectionH_);
  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> selectedTrackExtras( new reco::TrackExtraCollection() );
  std::auto_ptr<TrackingRecHitCollection> selectedTrackHits( new TrackingRecHitCollection() );

  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< dtSegmentCollectionH_->size()<<" DT segments.";
  unsigned int index_dt_segment = 0;
  for(DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
      segment != dtSegmentCollectionH_->end(); ++segment , index_dt_segment++) {
    LocalPoint  segmentLocalPosition       = segment->localPosition();
    LocalVector segmentLocalDirection      = segment->localDirection();
    LocalError  segmentLocalPositionError  = segment->localPositionError();
    LocalError  segmentLocalDirectionError = segment->localDirectionError();
    DetId geoid = segment->geographicalId();
    DTChamberId dtdetid = DTChamberId(geoid);
    int wheel = dtdetid.wheel();
    int station = dtdetid.station();
    int sector = dtdetid.sector();
    
    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x()/segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y()/segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("MuonTrackProducer") 
      <<"\nDT segment index :"<<index_dt_segment
      <<"\nchamber Wh:"<<wheel<<",St:"<<station<<",Se:"<<sector
      <<"\nLocal Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
      <<"Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
  }

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< cscSegmentCollectionH_->size()<<" CSC segments.";
  unsigned int index_csc_segment = 0;
  for(CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
      segment != cscSegmentCollectionH_->end(); ++segment , index_csc_segment++) {
    LocalPoint  segmentLocalPosition       = segment->localPosition();
    LocalVector segmentLocalDirection      = segment->localDirection();
    LocalError  segmentLocalPositionError  = segment->localPositionError();
    LocalError  segmentLocalDirectionError = segment->localDirectionError();

    DetId geoid = segment->geographicalId();
    CSCDetId cscdetid = CSCDetId(geoid);
    int endcap = cscdetid.endcap();
    int station = cscdetid.station();
    int ring = cscdetid.ring();
    int chamber = cscdetid.chamber(); 
    
    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x()/segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y()/segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("MuonTrackProducer") 
      <<"\nCSC segment index :"<<index_csc_segment
      <<"\nchamber Endcap:"<<endcap<<",St:"<<station<<",Ri:"<<ring<<",Ch:"<<chamber
      <<"\nLocal Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
      <<"Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
  }

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< muonCollectionH->size() <<" reco::Muons.";
  unsigned int muon_index = 0;
  for(reco::MuonCollection::const_iterator muon = muonCollectionH->begin();
       muon != muonCollectionH->end(); ++muon, muon_index++) {
    edm::LogVerbatim("MuonTrackProducer") <<"\n******* muon index : "<<muon_index;
    
    std::vector<bool> isGood;    
    for(unsigned int index=0; index<selectionTags.size(); ++index) {
      isGood.push_back(false);

      muon::SelectionType muonType = muon::selectionTypeFromString(selectionTags[index]);
      isGood[index] = muon::isGoodMuon(*muon, muonType);
    }

    bool isGoodResult=true;
    for(unsigned int index=0; index<isGood.size(); ++index) {
      edm::LogVerbatim("MuonTrackProducer") << "selectionTag = "<<selectionTags[index]<< ": "<<isGood[index]<<"\n";
      isGoodResult *= isGood[index];
    }

    if (isGoodResult) {
      // new copy of Track
      reco::TrackRef trackref;
      if (trackType == "innerTrack") {
        if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
        else continue;
      } 
      else if (trackType == "outerTrack") {
        if (muon->outerTrack().isNonnull()) trackref = muon->outerTrack();
        else continue;
      } 
      else if (trackType == "globalTrack") {
        if (muon->globalTrack().isNonnull()) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "innerTrackPlusSegments") {
	if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
	else continue;
      }

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      reco::Track* newTrk = new reco::Track(*trk);

      newTrk->setExtra( reco::TrackExtraRef( rTrackExtras, idx++ ) );
      PropagationDirection seedDir = trk->seedDirection();
      // new copy of track Extras
      reco::TrackExtra * newExtra = new reco::TrackExtra( trk->outerPosition(), trk->outerMomentum(), 
                                        trk->outerOk(), trk->innerPosition(), 
                                        trk->innerMomentum(), trk->innerOk(),
                                        trk->outerStateCovariance(), trk->outerDetId(),
                                        trk->innerStateCovariance(), trk->innerDetId() , seedDir ) ;

      // new copy of the silicon hits; add hit refs to Extra and hits to hit collection
      
      //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing initial hit_pattern";
      //      trk->hitPattern().print();
      unsigned int nHitsToAdd = 0;
      for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {
        TrackingRecHit* hit = (*iHit)->clone();
        selectedTrackHits->push_back( hit );
        ++nHitsToAdd;
      }
      newExtra->setHits( rHits, hidx, nHitsToAdd );
      hidx += nHitsToAdd;
      if (trackType == "innerTrackPlusSegments") { 
	
	int wheel, station, sector;
	int endcap, /*station, */ ring, chamber;
	
	edm::LogVerbatim("MuonTrackProducer") <<"Number of chambers: "<<muon->matches().size()
					      <<", arbitrated: "<<muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration);
	unsigned int index_chamber = 0;
	
	for(std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
	    chamberMatch != muon->matches().end(); ++chamberMatch, index_chamber++) {
	  std::stringstream chamberStr;
	  chamberStr <<"\nchamber index: "<<index_chamber; 
	  
	  int subdet = chamberMatch->detector();
	  DetId did = chamberMatch->id;
	  
	  if (subdet == MuonSubdetId::DT) {
	    DTChamberId dtdetid = DTChamberId(did);
	    wheel = dtdetid.wheel();
	    station = dtdetid.station();
	    sector = dtdetid.sector();
	    chamberStr << ", DT chamber Wh:"<<wheel<<",St:"<<station<<",Se:"<<sector;
	  } 
	  else if (subdet == MuonSubdetId::CSC) {
	    CSCDetId cscdetid = CSCDetId(did);
	    endcap = cscdetid.endcap();
	    station = cscdetid.station();
	    ring = cscdetid.ring();
	    chamber = cscdetid.chamber();
	    chamberStr << ", CSC chamber End:"<<endcap<<",St:"<<station<<",Ri:"<<ring<<",Ch:"<<chamber;
	  }
	  
	  chamberStr << ", Number of segments: "<<chamberMatch->segmentMatches.size();
	  edm::LogVerbatim("MuonTrackProducer") << chamberStr.str();

	  unsigned int index_segment = 0;
	  
	  for(std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
	      segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch, index_segment++) {
	    
	    float segmentX = segmentMatch->x;
	    float segmentY = segmentMatch->y ;
	    float segmentdXdZ = segmentMatch->dXdZ;
	    float segmentdYdZ = segmentMatch->dYdZ;
	    float segmentXerr = segmentMatch->xErr;
	    float segmentYerr = segmentMatch->yErr;
	    float segmentdXdZerr = segmentMatch->dXdZErr;
	    float segmentdYdZerr = segmentMatch->dYdZErr;
	    
	    CSCSegmentRef segmentCSC = segmentMatch->cscSegmentRef;
	    DTRecSegment4DRef segmentDT = segmentMatch->dtSegmentRef;
	    
	    bool segment_arbitrated_Ok = (segmentMatch->isMask(reco::MuonSegmentMatch::BestInChamberByDR) && 
					  segmentMatch->isMask(reco::MuonSegmentMatch::BelongsToTrackByDR));
	    
	    std::string ARBITRATED(" ***Arbitrated Off*** ");
	    if (segment_arbitrated_Ok) ARBITRATED = " ***ARBITRATED OK*** ";

	    if (subdet == MuonSubdetId::DT) {	      
	      edm::LogVerbatim("MuonTrackProducer")
		<<"\n\t segment index: "<<index_segment << ARBITRATED
		<<"\n\t  Local Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
		<<"\n\t  Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
	      
	      if (!segment_arbitrated_Ok) continue;
	      
	      if (segmentDT.get() != 0) {
		const DTRecSegment4D* segment = segmentDT.get();
		
		edm::LogVerbatim("MuonTrackProducer")<<"\t ===> MATCHING with DT segment with index = "<<segmentDT.key();
		
		if(segment->hasPhi()) {
		  const DTChamberRecSegment2D* phiSeg = segment->phiSegment();
		  std::vector<const TrackingRecHit*> phiHits = phiSeg->recHits();
                  unsigned int nHitsAdded = 0;
		  for(std::vector<const TrackingRecHit*>::const_iterator ihit = phiHits.begin();
		      ihit != phiHits.end(); ++ihit) {
		    TrackingRecHit* seghit = (*ihit)->clone();
             newTrk->appendHitPattern(*seghit);
		    //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		    //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		    selectedTrackHits->push_back( seghit );
                    ++nHitsAdded;
		  }
                  newExtra->setHits( rHits, hidx, nHitsAdded );
                  hidx += nHitsAdded;
		}
		
		if(segment->hasZed()) {
		  const DTSLRecSegment2D* zSeg = (*segment).zSegment();
		  std::vector<const TrackingRecHit*> zedHits = zSeg->recHits();
                  unsigned int nHitsAdded = 0;
		  for(std::vector<const TrackingRecHit*>::const_iterator ihit = zedHits.begin();
		      ihit != zedHits.end(); ++ihit) {
		    TrackingRecHit* seghit = (*ihit)->clone();
            newTrk->appendHitPattern(*seghit);
		    //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		    //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		    selectedTrackHits->push_back( seghit );
                    ++nHitsAdded;
		  }
                  newExtra->setHits( rHits, hidx, nHitsAdded );
                  hidx += nHitsAdded;
		}
	      } else edm::LogWarning("MuonTrackProducer")<<"\n***WARNING: UNMATCHED DT segment ! \n";
	    } // if (subdet == MuonSubdetId::DT)

	    else if (subdet == MuonSubdetId::CSC) {
	      edm::LogVerbatim("MuonTrackProducer")
		<<"\n\t segment index: "<<index_segment << ARBITRATED
		<<"\n\t  Local Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
		<<"\n\t  Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
	      
	      if (!segment_arbitrated_Ok) continue;
	      
	      if (segmentCSC.get() != 0) {
		const CSCSegment* segment = segmentCSC.get();
		
		edm::LogVerbatim("MuonTrackProducer")<<"\t ===> MATCHING with CSC segment with index = "<<segmentCSC.key();
		
		std::vector<const TrackingRecHit*> hits = segment->recHits();
                unsigned int nHitsAdded = 0;
		for(std::vector<const TrackingRecHit*>::const_iterator ihit = hits.begin();
		    ihit != hits.end(); ++ihit) {
		  TrackingRecHit* seghit = (*ihit)->clone();
          newTrk->appendHitPattern(*seghit);
		  //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		  //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		  selectedTrackHits->push_back( seghit );
                  ++nHitsAdded;
		}
                newExtra->setHits( rHits, hidx, nHitsAdded );
                hidx += nHitsAdded;
	      } else edm::LogWarning("MuonTrackProducer")<<"\n***WARNING: UNMATCHED CSC segment ! \n";
	    }  //  else if (subdet == MuonSubdetId::CSC)

	  } // loop on vector<MuonSegmentMatch>	  
	} // loop on vector<MuonChamberMatch>	
      } // if (trackType == "innerTrackPlusSegments")
      
      //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing final hit_pattern";
      //      newTrk->hitPattern().print();
      
      selectedTracks->push_back( *newTrk );
      selectedTrackExtras->push_back( *newExtra );

    } // if (isGoodResult)
  }  // loop on reco::MuonCollection
  
  iEvent.put(selectedTracks);
  iEvent.put(selectedTrackExtras);
  iEvent.put(selectedTrackHits);
}
