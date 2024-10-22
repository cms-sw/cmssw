#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "Validation/MuonME0Validation/interface/ME0SegmentsValidation.h"
#include <TMath.h>

ME0SegmentsValidation::ME0SegmentsValidation(const edm::ParameterSet &cfg) : ME0BaseValidation(cfg) {
  InputTagToken_Segments = consumes<ME0SegmentCollection>(cfg.getParameter<edm::InputTag>("segmentInputLabel"));
  InputTagToken_Digis = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagTokenST_ = consumes<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simInputLabelST"));
  sigma_x_ = cfg.getParameter<double>("sigma_x");
  sigma_y_ = cfg.getParameter<double>("sigma_y");
  eta_max_ = cfg.getParameter<double>("eta_max");
  eta_min_ = cfg.getParameter<double>("eta_min");
  pt_min_ = cfg.getParameter<double>("pt_min");
  isMuonGun_ = cfg.getParameter<bool>("isMuonGun");
}

void ME0SegmentsValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                           edm::Run const &Run,
                                           edm::EventSetup const &iSetup) {
  LogDebug("MuonME0SegmentsValidation") << "Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0RecHitsV/ME0SegmentsTask");

  unsigned int nregion = 2;

  edm::LogInfo("MuonME0SegmentsValidation") << "+++ Info : # of region : " << nregion << std::endl;

  LogDebug("MuonME0SegmentsValidation") << "+++ Info : finish to get geometry information from ES.\n";

  me0_simsegment_eta = ibooker.book1D("me0_simsegment_eta", "SimSegment Eta Distribution; #eta; entries", 8, 2.0, 2.8);
  me0_simsegment_pt = ibooker.book1D("me0_simsegment_pt", "SimSegment pT Distribution; p_{T}; entries", 20, 0.0, 100.0);
  me0_simsegment_phi =
      ibooker.book1D("me0_simsegment_phi", "SimSegments phi Distribution; #phi; entries", 18, -M_PI, +M_PI);

  me0_matchedsimsegment_eta =
      ibooker.book1D("me0_matchedsimsegment_eta", "Matched SimSegment Eta Distribution; #eta; entries", 8, 2.0, 2.8);
  me0_matchedsimsegment_pt =
      ibooker.book1D("me0_matchedsimsegment_pt", "Matched SimSegment pT Distribution; p_{T}; entries", 20, 0.0, 100.0);
  me0_matchedsimsegment_phi = ibooker.book1D(
      "me0_matchedsimsegment_phi", "Matched SimSegments phi Distribution; #phi; entries", 18, -M_PI, +M_PI);

  me0_segment_chi2 = ibooker.book1D("me0_seg_Chi2", "#chi^{2}; #chi^{2}; # Segments", 100, 0, 100);
  me0_segment_redchi2 = ibooker.book1D("me0_seg_ReducedChi2", "#chi^{2}/ndof; #chi^{2}/ndof; # Segments", 100, 0, 5);
  me0_segment_ndof = ibooker.book1D("me0_seg_ndof", "ndof; ndof; #Segments", 50, 0, 50);
  me0_segment_numRH =
      ibooker.book1D("me0_seg_NumberRH", "Number of fitted RecHits; # RecHits; entries", 11, -0.5, 10.5);
  me0_segment_numRHSig =
      ibooker.book1D("me0_seg_NumberRHSig", "Number of fitted Signal RecHits; # RecHits; entries", 11, -0.5, 10.5);
  me0_segment_numRHBkg =
      ibooker.book1D("me0_seg_NumberRHBkg", "Number of fitted BKG RecHits; # RecHits; entries", 11, -0.5, 10.5);
  // me0_segment_EtaRH   = ibooker.book1D("me0_specRH_globalEta","Fitted RecHits
  // Eta Distribution; #eta; entries",200,-4.0,4.0); me0_segment_PhiRH   =
  // ibooker.book1D("me0_specRH_globalPhi","Fitted RecHits Phi Distribution;
  // #eta; entries",18,-3.14,3.14);
  me0_segment_time = ibooker.book1D("me0_seg_time", "Segment Timing; ns; entries", 300, -150, 150);
  me0_segment_timeErr = ibooker.book1D("me0_seg_timErr", "Segment Timing Error; ns; entries", 50, 0, 0.5);
  me0_segment_size =
      ibooker.book1D("me0_seg_size", "Segment Multiplicity; Number of ME0 segments; entries", 200, 0, 200);

  for (unsigned int region_num = 0; region_num < nregion; region_num++) {
    me0_specRH_zr[region_num] = BookHistZR(ibooker, "me0_specRH_tot", "Segment RecHits", region_num);
    for (unsigned int layer_num = 0; layer_num < 6; layer_num++) {
      // me0_strip_dg_zr[region_num][layer_num] =
      // BookHistZR(ibooker,"me0_strip_dg","SimHit",region_num,layer_num);
      me0_specRH_xy[region_num][layer_num] =
          BookHistXY(ibooker, "me0_specRH", "Segment RecHits", region_num, layer_num);
      // me0_rh_xy_Muon[region_num][layer_num] =
      // BookHistXY(ibooker,"me0_rh","RecHit Muon",region_num,layer_num);

      std::string histo_name_DeltaX =
          std::string("me0_specRH_DeltaX_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_name_DeltaY =
          std::string("me0_specRH_DeltaY_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_label_DeltaX = "Segment RecHits Delta X : region" + regionLabel[region_num] + " layer " +
                                       layerLabel[layer_num] + " " + " ; x_{SimHit} - x_{Segment RecHits} ; entries";
      std::string histo_label_DeltaY = "Segment RecHits Delta Y : region" + regionLabel[region_num] + " layer " +
                                       layerLabel[layer_num] + " " + " ; y_{SimHit} - y_{Segment RecHit} ; entries";

      me0_specRH_DeltaX[region_num][layer_num] =
          ibooker.book1D(histo_name_DeltaX.c_str(), histo_label_DeltaX.c_str(), 100, -10, 10);
      me0_specRH_DeltaY[region_num][layer_num] =
          ibooker.book1D(histo_name_DeltaY.c_str(), histo_label_DeltaY.c_str(), 100, -10, 10);

      std::string histo_name_PullX =
          std::string("me0_specRH_PullX_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_name_PullY =
          std::string("me0_specRH_PullY_r") + regionLabel[region_num] + "_l" + layerLabel[layer_num];
      std::string histo_label_PullX = "Segment RecHits Pull X : region" + regionLabel[region_num] + " layer " +
                                      layerLabel[layer_num] + " " +
                                      " ; #frac{x_{SimHit} - x_{Segment "
                                      "RecHit}}{#sigma_{x,RecHit}} ; entries";
      std::string histo_label_PullY = "Segment RecHits Pull Y : region" + regionLabel[region_num] + " layer " +
                                      layerLabel[layer_num] + " " +
                                      " ; #frac{y_{SimHit} - y_{Segment "
                                      "RecHit}}{#sigma_{y,RecHit}} ; entries";

      me0_specRH_PullX[region_num][layer_num] =
          ibooker.book1D(histo_name_PullX.c_str(), histo_label_DeltaX.c_str(), 100, -10, 10);
      me0_specRH_PullY[region_num][layer_num] =
          ibooker.book1D(histo_name_PullY.c_str(), histo_label_DeltaY.c_str(), 100, -10, 10);
    }
  }
}

ME0SegmentsValidation::~ME0SegmentsValidation() {}

void ME0SegmentsValidation::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const ME0Geometry *ME0Geometry_ = &iSetup.getData(geomToken_);

  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);

  edm::Handle<edm::SimTrackContainer> simTracks;
  e.getByToken(InputTagTokenST_, simTracks);

  edm::Handle<ME0SegmentCollection> ME0Segments;
  e.getByToken(InputTagToken_Segments, ME0Segments);

  edm::Handle<ME0DigiPreRecoCollection> ME0Digis;
  e.getByToken(InputTagToken_Digis, ME0Digis);

  if (!ME0Digis.isValid()) {
    edm::LogError("ME0SegmentsValidation") << "Cannot get ME0Digis by Token InputTagToken";
    return;
  }

  if (!ME0Segments.isValid()) {
    edm::LogError("ME0SegmentsValidation") << "Cannot get ME0RecHits/ME0Segments by Token InputTagToken";
    return;
  }

  if (!ME0Hits.isValid()) {
    edm::LogError("ME0HitsValidation") << "Cannot get ME0Hits by Token simInputTagToken";
    return;
  }

  MapTypeSim myMap;
  MapTypeSeg myMapSeg;

  edm::SimTrackContainer::const_iterator simTrack;
  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack) {
    edm::PSimHitContainer selectedME0Hits;

    if (!isSimTrackGood(simTrack))
      continue;

    for (edm::PSimHitContainer::const_iterator itHit = ME0Hits->begin(); itHit != ME0Hits->end(); ++itHit) {
      int particleType_sh = itHit->particleType();
      int evtId_sh = itHit->eventId().event();
      int bx_sh = itHit->eventId().bunchCrossing();
      int procType_sh = itHit->processType();
      if (!(abs(particleType_sh) == 13 && evtId_sh == 0 && bx_sh == 0 && procType_sh == 0))
        continue;

      if (isSimMatched(simTrack, itHit)) {
        selectedME0Hits.push_back(*itHit);
        ;
      }

    }  // End loop SHs

    if (selectedME0Hits.size() >= 3) {
      myMap.insert(MapTypeSim::value_type(simTrack, selectedME0Hits));
      me0_simsegment_eta->Fill(std::abs((*simTrack).momentum().eta()));
      me0_simsegment_pt->Fill((*simTrack).momentum().pt());
      me0_simsegment_phi->Fill((*simTrack).momentum().phi());
    }
  }

  me0_segment_size->Fill(ME0Segments->size());

  for (auto me0s = ME0Segments->begin(); me0s != ME0Segments->end(); me0s++) {
    // The ME0 Ensamble DetId refers to layer = 1
    ME0DetId id = me0s->me0DetId();
    auto chamber = ME0Geometry_->chamber(id);
    auto segLP = me0s->localPosition();
    auto segLD = me0s->localDirection();
    auto me0rhs = me0s->specificRecHits();

    //   float localX = segLP.x();
    //   float localY = segLP.y();
    //   float dirTheta = segLD.theta();
    //   float dirPhi = segLD.phi();
    int numberRH = me0rhs.size();
    float chi2 = (float)me0s->chi2();
    float ndof = me0s->degreesOfFreedom();
    double time = me0s->time();
    double timeErr = me0s->timeErr();

    float reducedChi2 = chi2 / ndof;

    me0_segment_chi2->Fill(chi2);
    me0_segment_redchi2->Fill(reducedChi2);
    me0_segment_ndof->Fill(ndof);
    me0_segment_numRH->Fill(numberRH);

    me0_segment_time->Fill(time);
    me0_segment_timeErr->Fill(timeErr);

    int numberRHSig = 0;
    int numberRHBkg = 0;
    std::vector<ME0RecHit> selectedME0RecHits;

    for (auto rh = me0rhs.begin(); rh != me0rhs.end(); rh++) {
      auto me0id = rh->me0Id();
      auto rhr = ME0Geometry_->etaPartition(me0id);
      auto rhLP = rh->localPosition();

      auto result = isMatched(me0id, rhLP, ME0Digis);
      if (result.second == 1) {
        ++numberRHSig;
        selectedME0RecHits.push_back(*rh);

      } else
        ++numberRHBkg;

      auto erhLEP = rh->localPositionError();
      auto rhGP = rhr->toGlobal(rhLP);
      auto rhLPSegm = chamber->toLocal(rhGP);
      float xe = segLP.x() + segLD.x() * rhLPSegm.z() / segLD.z();
      float ye = segLP.y() + segLD.y() * rhLPSegm.z() / segLD.z();
      float ze = rhLPSegm.z();
      LocalPoint extrPoint(xe, ye, ze);                           // in segment rest frame
      auto extSegm = rhr->toLocal(chamber->toGlobal(extrPoint));  // in layer restframe

      int region = me0id.region();
      int layer = me0id.layer();
      //     int chamber = me0id.chamber();

      float x = rhLP.x();
      float xErr = erhLEP.xx();
      float y = rhLP.y();
      float yErr = erhLEP.yy();

      float globalR = rhGP.perp();
      float globalX = rhGP.x();
      float globalY = rhGP.y();
      float globalZ = rhGP.z();

      float xExt = extSegm.x();
      float yExt = extSegm.y();

      float pull_x = (x - xExt) / sqrt(xErr);
      float pull_y = (y - yExt) / sqrt(yErr);

      int region_num = 0;
      if (region == -1)
        region_num = 0;
      else if (region == 1)
        region_num = 1;
      int layer_num = layer - 1;

      me0_specRH_xy[region_num][layer_num]->Fill(globalX, globalY);
      me0_specRH_zr[region_num]->Fill(globalZ, globalR);

      me0_specRH_DeltaX[region_num][layer_num]->Fill(x - xExt);
      me0_specRH_DeltaY[region_num][layer_num]->Fill(y - yExt);
      me0_specRH_PullX[region_num][layer_num]->Fill(pull_x);
      me0_specRH_PullY[region_num][layer_num]->Fill(pull_y);
    }

    me0_segment_numRHSig->Fill(numberRHSig);
    me0_segment_numRHBkg->Fill(numberRHBkg);
    myMapSeg.insert(MapTypeSeg::value_type(me0s, selectedME0RecHits));
  }

  //------------------- SimToReco -------------------

  for (auto const &st : myMap) {  // loop over the signal simTracks

    int num_sh = st.second.size();
    bool isThereOneSegmentMatched = false;

    for (auto const &seg : myMapSeg) {  // loop over the reconstructed me0 segments

      int num_sh_matched = 0;
      if (seg.second.empty())
        continue;

      for (auto const &sh : st.second) {  // loop over the me0 simHits left by
                                          // the signal simTracks

        for (auto const &rh : seg.second) {  // loop over the tracking recHits
                                             // already matched to signal digis

          auto me0id = rh.me0Id();
          int region_rh = (int)me0id.region();
          int layer_rh = (int)me0id.layer();
          int chamber_rh = (int)me0id.chamber();
          int roll_rh = (int)me0id.roll();

          const ME0DetId id(sh.detUnitId());
          int region_sh = id.region();
          int layer_sh = id.layer();
          int chamber_sh = id.chamber();
          int roll_sh = id.roll();

          if (!(region_sh == region_rh && chamber_sh == chamber_rh && layer_sh == layer_rh && roll_sh == roll_rh))
            continue;

          LocalPoint lp_sh = sh.localPosition();
          LocalPoint lp_rh = rh.localPosition();

          GlobalPoint gp_sh = ME0Geometry_->idToDet(id)->surface().toGlobal(lp_sh);
          GlobalPoint gp = ME0Geometry_->idToDet((rh).me0Id())->surface().toGlobal(lp_rh);
          float dphi_glob = gp_sh.phi() - gp.phi();
          float deta_glob = gp_sh.eta() - gp.eta();

          if (fabs(dphi_glob) < 3 * sigma_x_ && fabs(deta_glob) < 3 * sigma_y_)
            ++num_sh_matched;

        }  // End loop over RHs

      }  // End loop over SHs

      float quality = 0;
      if (num_sh != 0)
        quality = num_sh_matched / (1.0 * num_sh);
      if (quality > 0)
        isThereOneSegmentMatched = true;

    }  // End loop over segments

    // Fill hsitograms
    if (isThereOneSegmentMatched) {
      me0_matchedsimsegment_eta->Fill(std::abs((*(st.first)).momentum().eta()));
      me0_matchedsimsegment_pt->Fill((*(st.first)).momentum().pt());
      me0_matchedsimsegment_phi->Fill((*(st.first)).momentum().phi());
    }

  }  // End loop over STs
}

std::pair<int, int> ME0SegmentsValidation::isMatched(ME0DetId me0id,
                                                     LocalPoint rhLP,
                                                     edm::Handle<ME0DigiPreRecoCollection> ME0Digis) {
  int region_rh = (int)me0id.region();
  int layer_rh = (int)me0id.layer();
  int roll_rh = (int)me0id.roll();
  int chamber_rh = (int)me0id.chamber();

  float l_x_rh = rhLP.x();
  float l_y_rh = rhLP.y();

  int particleType = 0;
  int isPrompt = -1;

  for (ME0DigiPreRecoCollection::DigiRangeIterator cItr = ME0Digis->begin(); cItr != ME0Digis->end(); cItr++) {
    ME0DetId id = (*cItr).first;

    int region_dg = (int)id.region();
    int layer_dg = (int)id.layer();
    int roll_dg = (int)id.roll();
    int chamber_dg = (int)id.chamber();

    if (region_rh != region_dg)
      continue;
    if (layer_rh != layer_dg)
      continue;
    if (chamber_rh != chamber_dg)
      continue;
    if (roll_rh != roll_dg)
      continue;

    ME0DigiPreRecoCollection::const_iterator digiItr;
    for (digiItr = (*cItr).second.first; digiItr != (*cItr).second.second; ++digiItr) {
      float l_x_dg = digiItr->x();
      float l_y_dg = digiItr->y();

      if (l_x_rh != l_x_dg)
        continue;
      if (l_y_rh != l_y_dg)
        continue;

      particleType = digiItr->pdgid();
      isPrompt = digiItr->prompt();
    }
  }

  std::pair<int, int> result;
  result = std::make_pair(particleType, isPrompt);

  return result;
}

bool ME0SegmentsValidation::isSimTrackGood(edm::SimTrackContainer::const_iterator t) {
  if ((*t).noVertex() && !isMuonGun_)
    return false;
  if ((*t).noGenpart() && !isMuonGun_)
    return false;
  if (std::abs((*t).type()) != 13)
    return false;  // only interested in direct muon simtracks
  if ((*t).momentum().pt() < pt_min_)
    return false;
  const float eta(std::abs((*t).momentum().eta()));
  if (eta < eta_min_ || eta > eta_max_)
    return false;  // no GEMs could be in such eta
  return true;
}

bool ME0SegmentsValidation::isSimMatched(edm::SimTrackContainer::const_iterator simTrack,
                                         edm::PSimHitContainer::const_iterator itHit) {
  bool result = false;
  int trackId = simTrack->trackId();
  int trackId_sim = itHit->trackId();
  if (trackId == trackId_sim)
    result = true;
  return result;
}
