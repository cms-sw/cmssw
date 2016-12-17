#include "Validation/MuonME0Validation/interface/ME0SegmentsValidation.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include <TMath.h>

ME0SegmentsValidation::ME0SegmentsValidation(const edm::ParameterSet& cfg):  ME0BaseValidation(cfg)
{
    InputTagToken_Segments = consumes<ME0SegmentCollection>(cfg.getParameter<edm::InputTag>("segmentInputLabel"));
    InputTagToken_Digis = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
}

void ME0SegmentsValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
    //edm::ESHandle<ME0Geometry> hGeom;
    //iSetup.get<MuonGeometryRecord>().get(hGeom);
    //const ME0Geometry* ME0Geometry_ =( &*hGeom);
    
    LogDebug("MuonME0SegmentsValidation")<<"Info : Loading Geometry information\n";
    ibooker.setCurrentFolder("MuonME0RecHitsV/ME0SegmentsTask");
    
    unsigned int nregion  = 2;
    
    edm::LogInfo("MuonME0SegmentsValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;
    
    LogDebug("MuonME0SegmentsValidation")<<"+++ Info : finish to get geometry information from ES.\n";
    
    me0_segment_chi2     = ibooker.book1D("me0_seg_Chi2","#chi^{2}; #chi^{2}; # Segments",100,0,100);
    me0_segment_redchi2  = ibooker.book1D("me0_seg_ReducedChi2","#chi^{2}/ndof; #chi^{2}/ndof; # Segments",100,0,5);
    me0_segment_ndof     = ibooker.book1D("me0_seg_ndof","ndof; ndof; #Segments",100,0,100);
    me0_segment_numRH    = ibooker.book1D("me0_seg_NumberRH","Number of fitted RecHits; # RecHits; entries",11,-0.5,10.5);
    me0_segment_numRHSig = ibooker.book1D("me0_seg_NumberRHSig","Number of fitted Signal RecHits; # RecHits; entries",11,-0.5,10.5);
    me0_segment_numRHBkg = ibooker.book1D("me0_seg_NumberRHBkg","Number of fitted BKG RecHits; # RecHits; entries",11,-0.5,10.5);
    //me0_segment_EtaRH   = ibooker.book1D("me0_specRH_globalEta","Fitted RecHits Eta Distribution; #eta; entries",200,-4.0,4.0);
    //me0_segment_PhiRH   = ibooker.book1D("me0_specRH_globalPhi","Fitted RecHits Phi Distribution; #eta; entries",18,-3.14,3.14);
    me0_segment_time     = ibooker.book1D("me0_seg_time","Segment Timing; ns; entries",300,-150,150);
    me0_segment_timeErr  = ibooker.book1D("me0_seg_timErr","Segment Timing Error; ns; entries",50,0,0.5);
    me0_segment_size     = ibooker.book1D("me0_seg_size","Segment Multiplicity; Number of ME0 segments; entries",200,0,200);
    
    for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
        me0_specRH_zr[region_num] = BookHistZR(ibooker,"me0_specRH_tot","Segment RecHits",region_num);
        for( unsigned int layer_num = 0 ; layer_num < 6 ; layer_num++) {
            
          //me0_strip_dg_zr[region_num][layer_num] = BookHistZR(ibooker,"me0_strip_dg","SimHit",region_num,layer_num);
          me0_specRH_xy[region_num][layer_num] = BookHistXY(ibooker,"me0_specRH","Segment RecHits",region_num,layer_num);
          //me0_rh_xy_Muon[region_num][layer_num] = BookHistXY(ibooker,"me0_rh","RecHit Muon",region_num,layer_num);

          std::string histo_name_DeltaX = std::string("me0_specRH_DeltaX_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_name_DeltaY = std::string("me0_specRH_DeltaY_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_label_DeltaX = "Segment RecHits Delta X : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; x_{SimHit} - x_{Segment RecHits} ; entries";
          std::string histo_label_DeltaY = "Segment RecHits Delta Y : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; y_{SimHit} - y_{Segment RecHit} ; entries";

          me0_specRH_DeltaX[region_num][layer_num] = ibooker.book1D(histo_name_DeltaX.c_str(), histo_label_DeltaX.c_str(),100,-10,10);
          me0_specRH_DeltaY[region_num][layer_num] = ibooker.book1D(histo_name_DeltaY.c_str(), histo_label_DeltaY.c_str(),100,-10,10);

          std::string histo_name_PullX = std::string("me0_specRH_PullX_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_name_PullY = std::string("me0_specRH_PullY_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_label_PullX = "Segment RecHits Pull X : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; #frac{x_{SimHit} - x_{Segment RecHit}}{#sigma_{x,RecHit}} ; entries";
          std::string histo_label_PullY = "Segment RecHits Pull Y : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; #frac{y_{SimHit} - y_{Segment RecHit}}{#sigma_{y,RecHit}} ; entries";

          me0_specRH_PullX[region_num][layer_num] = ibooker.book1D(histo_name_PullX.c_str(), histo_label_DeltaX.c_str(),100,-10,10);
          me0_specRH_PullY[region_num][layer_num] = ibooker.book1D(histo_name_PullY.c_str(), histo_label_DeltaY.c_str(),100,-10,10);
            
        }
    }
}


ME0SegmentsValidation::~ME0SegmentsValidation() {
}


void ME0SegmentsValidation::analyze(const edm::Event& e,
                                    const edm::EventSetup& iSetup)
{
    edm::ESHandle<ME0Geometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    const ME0Geometry* ME0Geometry_ =( &*hGeom);
    
    edm::Handle<ME0SegmentCollection> ME0Segments;
    e.getByToken(InputTagToken_Segments, ME0Segments);
    
    edm::Handle<ME0DigiPreRecoCollection> ME0Digis;
    e.getByToken(InputTagToken_Digis, ME0Digis);
    
    if (!ME0Digis.isValid() ) {
        edm::LogError("ME0SegmentsValidation") << "Cannot get ME0Digis by Token InputTagToken";
        return ;
    }
    
    if (!ME0Segments.isValid() ) {
        edm::LogError("ME0SegmentsValidation") << "Cannot get ME0RecHits/ME0Segments by Token InputTagToken";
        return ;
    }
    
    me0_segment_size->Fill(ME0Segments->size());
    
    for (auto me0s = ME0Segments->begin(); me0s != ME0Segments->end(); me0s++) {
        
        // The ME0 Ensamble DetId refers to layer = 1
        ME0DetId id = me0s->me0DetId();
        auto chamber = ME0Geometry_->chamber(id);
        auto segLP = me0s->localPosition();
        auto segLD = me0s->localDirection();
        auto me0rhs = me0s->specificRecHits();
        
        //   Float_t localX = segLP.x();
        //   Float_t localY = segLP.y();
        //   Float_t dirTheta = segLD.theta();
        //   Float_t dirPhi = segLD.phi();
        Short_t numberRH = me0rhs.size();
        Float_t chi2 = (Float_t) me0s->chi2();
        Float_t ndof = me0s->degreesOfFreedom();
        Double_t time = me0s->time();
        Double_t timeErr = me0s->timeErr();
        
        Float_t reducedChi2 = chi2/ndof;
        
        me0_segment_chi2->Fill(chi2);
        me0_segment_redchi2->Fill(reducedChi2);
        me0_segment_ndof->Fill(ndof);
        me0_segment_numRH->Fill(numberRH);

        me0_segment_time->Fill(time);
        me0_segment_timeErr->Fill(timeErr);
     
        Short_t numberRHSig = 0;
        Short_t numberRHBkg = 0;
        
        for (auto rh = me0rhs.begin(); rh!= me0rhs.end(); rh++)
        {
            
            auto me0id = rh->me0Id();
            auto rhr = ME0Geometry_->etaPartition(me0id);
            auto rhLP = rh->localPosition();
            
            auto result = isMatched(me0id, rhLP, ME0Digis);
            if(result.second == 1) ++numberRHSig;
            else ++numberRHBkg;
            
            auto erhLEP = rh->localPositionError();
            auto rhGP = rhr->toGlobal(rhLP);
            auto rhLPSegm = chamber->toLocal(rhGP);
            float xe = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
            float ye = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
            float ze = rhLPSegm.z();
            LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
            auto extSegm = rhr->toLocal(chamber->toGlobal(extrPoint)); // in layer restframe
            
            Short_t region = me0id.region();
            Short_t layer = me0id.layer();
            //     Short_t chamber = me0id.chamber();
            
            Float_t x = rhLP.x();
            Float_t xErr = erhLEP.xx();
            Float_t y = rhLP.y();
            Float_t yErr = erhLEP.yy();
            
            Float_t globalR = rhGP.perp();
            Float_t globalX = rhGP.x();
            Float_t globalY = rhGP.y();
            Float_t globalZ = rhGP.z();
            
            Float_t xExt = extSegm.x();
            Float_t yExt = extSegm.y();
            
            Float_t pull_x = (x - xExt) / sqrt(xErr);
            Float_t pull_y = (y - yExt) / sqrt(yErr);
            
            int region_num=0 ;
            if ( region ==-1 ) region_num = 0 ;
            else if ( region==1) region_num = 1;
            int layer_num = layer-1;
            
            me0_specRH_xy[region_num][layer_num]->Fill(globalX,globalY);
            me0_specRH_zr[region_num]->Fill(globalZ,globalR);
            
            me0_specRH_DeltaX[region_num][layer_num]->Fill(x - xExt);
            me0_specRH_DeltaY[region_num][layer_num]->Fill(y - yExt);
            me0_specRH_PullX[region_num][layer_num]->Fill(pull_x);
            me0_specRH_PullY[region_num][layer_num]->Fill(pull_y);

        }
        
        me0_segment_numRHSig->Fill(numberRHSig);
        me0_segment_numRHBkg->Fill(numberRHBkg);
        
    }
    
}

std::pair<int,int> ME0SegmentsValidation::isMatched(auto me0id, auto rhLP, auto ME0Digis)
{
    Short_t region_rh = (Short_t) me0id.region();
    Short_t layer_rh = (Short_t) me0id.layer();
    Short_t roll_rh = (Short_t) me0id.roll();
    Short_t chamber_rh = (Short_t) me0id.chamber();
    
    Float_t l_x_rh = rhLP.x();
    Float_t l_y_rh = rhLP.y();
    
    Short_t particleType = 0;
    Short_t isPrompt = -1;
    
    for (ME0DigiPreRecoCollection::DigiRangeIterator cItr=ME0Digis->begin(); cItr!=ME0Digis->end(); cItr++) {
        
        ME0DetId id = (*cItr).first;
        
        Short_t region_dg = (Short_t) id.region();
        Short_t layer_dg = (Short_t) id.layer();
        Short_t roll_dg = (Short_t) id.roll();
        Short_t chamber_dg = (Short_t) id.chamber();
        
        if(region_rh != region_dg) continue;
        if(layer_rh != layer_dg) continue;
        if(chamber_rh != chamber_dg) continue;
        if(roll_rh != roll_dg) continue;
        
        ME0DigiPreRecoCollection::const_iterator digiItr;
        for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
        {
            
            Float_t l_x_dg = digiItr->x();
            Float_t l_y_dg = digiItr->y();
            
            if(l_x_rh != l_x_dg) continue;
            if(l_y_rh != l_y_dg) continue;
            
            particleType = digiItr->pdgid();
            isPrompt = digiItr->prompt();
            
        }
        
    }
    
    std::pair<int,int> result;
    result = std::make_pair(particleType,isPrompt);
    
    return result;
    
}
//
//bool ME0SegmentsValidation::isSimTrackGood(const SimTrack &t)
//{
//    
//    // SimTrack selection
//    if (t.noVertex()) return false;
//    if (t.noGenpart()) return false;
//    if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
////    if (t.momentum().pt() < minPt_ ) return false;
//    const float eta(std::abs(t.momentum().eta()));
//    if (eta > 2.0 || eta < 2.8 ) return false; // no GEMs could be in such eta
//    return true;
//    
//}
//
//
//bool isSimMatched(SimTrackContainer::const_iterator simTrack, edm::PSimHitContainer::const_iterator itHit)
//{
//    
//    bool result = false;
//    int trackId = simTrack->trackId();
//    int trackId_sim = itHit->trackId();
//    if(trackId == trackId_sim) result = true;
//    return result;
//    
//}
//
//edm::PSimHitContainer isTrackMatched(SimTrackContainer::const_iterator simTrack, const Event & event, const EventSetup& eventSetup)
//{
//    
//    edm::PSimHitContainer selectedGEMHits;
//    
//    edm::Handle<edm::PSimHitContainer> ME0Hits;
//    event.getByLabel(edm::InputTag("g4SimHits","MuonME0Hits"), ME0Hits);
//    
//    ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
//    eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
//    
//    for (edm::PSimHitContainer::const_iterator itHit = ME0Hits->begin(); itHit != ME0Hits->end(); ++itHit){
//							 
//        DetId id = DetId(itHit->detUnitId());
//        if (!(id.subdetId() == MuonSubdetId::ME0)) continue;
//        if(itHit->particleType() != (*simTrack).type()) continue;
//        
//        bool result = isSimMatched(simTrack, itHit);
//        if(result) selectedGEMHits.push_back(*itHit);
//        
//    }
//    
//    //std::cout<<"Size: "<<selectedGEMHits.size()<<std::endl;
//    return selectedGEMHits;
//    
//}


