#ifndef HISTOS_H
#define HISTOS_H

//___________________________________________________ PAIR
// ##### reco2sim
TH2D* phPtY_pairReco;                 // all dimu: pt vs y
TH2D* phPtY_pairRecoSim;              // matched dimuons: pt vs y

TH2D* phPtMinv_pairReco;              // all dimu: pt vs inv mass 
TH2D* phPtMinv_pairRecoSim;           // matched dimuons: pt vs inv mass 
 
TH2D* phPtY_pairRecoFake;             // pt vs eta fake pairs
TH2D* phPtMinv_pairRecoFake;          // pt vs eta fake pairs

TH2D* phMinvIdParent_pairRecoFake;    // pt vs eta fake pairs
TH2D* phMinvId_pairRecoFake;          // pt vs eta fake pairs

TH2D* phPtMinv_pairUnmatched;         // pair unmatched

TH2D *phIdParentIdParent_ff_ff;
TH2D *phIdId_ff_ff;

TH1D *phIdParent_muz0_muf;
TH2D *phIdIdParent_muz0_ff;

TH1D *phMinv_muf_muf;
TH2D *phIdParentIdParent_muf_muf;
TH2D *phPtIdParent_muf_muf;

TH1D *phMinv_muf_ff;
TH2D *phIdIdParentMuf_muf_ff;
TH2D *phPtIdParentMuf_muf_ff;	  
TH2D *phIdIdParentFf_muf_ff;

TH1D *phMinv_pairRecoFake;
TH1D *phMinv_muz0_muf;
TH1D *phMinv_muz0_ff;
TH1D *phMinv_ff_ff;


// semiFake dimuons (just one is fake)
TH2D* phPtY_pairSemiFake;         // pt vs eta dimuons, for dimuons with one fake muon and one true (from Z0)
TH2D* phTypeType_pairSemiFakeMuMu;// type of muons that get into the semiFake dimuons

// ##### sim2reco
TH2D* phPtY_pairSim;             // all sim dimuons: pt vs y 
TH2D* phPtY_pairSimReco;         // matched sim dimuons: pt vs y
TH2D* phPtY_pairSimLost;         // matched sim dimuons: pt vs y

TH2D* phPtMinv_pairSim;          // all sim dimuons: pt vs inv mass 
TH2D* phPtMinv_pairSimReco;      // matched sim dimuons: pt vs inv mass 
TH2D* phPtMinv_pairSimLost;      // all sim dimuons: pt vs inv mass 

TH2D* phPtY_pairResolution;
TH2D* phPtMinv_pairResolution;
TH2D* phPtPt_pairResolution;

//____________________________________________________SINGLE
// ##### reco2sim 
TH2D* phPtCharge_trkRecoSim ;       // charge recoTrk * simTrk charge vs pt of reco track
TH2D* phPtEta_trkReco ;             // all reco tracks
TH2D* phPtEta_trkRecoFake ;         // reco fake trk
TH2D* phPtEta_trkRecoMany ;         // reco trk, matched to more than 1 sim
TH2D* phPtEta_trkRecoSim ;          // reco trck matched to just 1 sim

TH2D* phPtId_trkRecoSim;            // ptrecoTrk vs id of sim track
TH2D* phPtIdparent_trkRecoSim;      // pt-reco vs id of sim parent

TH2D* phPtId_trkRecoFake;            // ptrecoTrk vs id of sim track
TH2D* phPtIdParent_trkRecoFake;      // pt-reco vs id of sim parent

// ##### sim2reco
TH2D* phPtCharge_trkSimReco;
TH2D* phIdId_simTrkParent; 

TH2D* phPtEta_trkSim;
TH2D* phPtEta_trkSimLost;
TH2D* phPtEta_trkSimMany ;
TH2D* phPtEta_trkSimReco;

TH2D* phPtEta_trkResolution;
TH2D* phPtPt_trkResolution;


#endif
