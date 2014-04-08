#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisGT.h"
#include "stdint.h"


L1Analysis::L1AnalysisGT::L1AnalysisGT()  
{
}

L1Analysis::L1AnalysisGT::~L1AnalysisGT()
{
}


void L1Analysis::L1AnalysisGT::SetEvm(const L1GlobalTriggerEvmReadoutRecord* gtevmrr)
{
    L1TcsWord tcsw = gtevmrr->tcsWord();

    //bx = tcsw.bxNr();
    //lumi = tcsw.luminositySegmentNr();
    //runn = tcsw.partRunNr();
    //eventn = tcsw.partTrigNr();
    //orbitn = tcsw.orbitNr();
    gt_.partrig_tcs = tcsw.partTrigNr();
    L1GtfeExtWord myGtfeExtWord = gtevmrr->gtfeWord();
    uint64_t gpsTime = myGtfeExtWord.gpsTime();

    gt_.gpsTimelo              = gpsTime&0xffffffff;
    gt_.gpsTimehi              = (gpsTime>>32)&0xffffffff;
    gt_.bstMasterStatus        = 0xffff & (myGtfeExtWord.bstMasterStatus()); 
    gt_.bstturnCountNumber     = myGtfeExtWord.turnCountNumber();
    gt_.bstlhcFillNumber       = myGtfeExtWord.lhcFillNumber();
    gt_.bstbeamMode            = 0xffff & (myGtfeExtWord.beamMode());   
    gt_.bstparticleTypeBeam1   = 0xffff & (myGtfeExtWord.particleTypeBeam1());
    gt_.bstparticleTypeBeam2   = 0xffff & (myGtfeExtWord.particleTypeBeam2());
    gt_.bstbeamMomentum        = 0xffff & (myGtfeExtWord.beamMomentum());
    gt_.bsttotalIntensityBeam1 = myGtfeExtWord.totalIntensityBeam1();
    gt_.bsttotalIntensityBeam2 = myGtfeExtWord.totalIntensityBeam2();
}

void L1Analysis::L1AnalysisGT::Set(const L1GlobalTriggerReadoutRecord* gtrr)
{
   
      for (int ibx=-1; ibx<=1; ibx++) {
      const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
      const L1GtPsbWord psb2 = gtrr->gtPsbWord(0xbb0e, ibx);


// ------ ETT, ETM, HTT and HTM from PSB14:

if (ibx == 0) {

  int psb_ett = psb2.aData(4);
  int ett_rank = psb_ett & 0xfff;
  gt_.RankETT = ett_rank;
  gt_.OvETT   = (psb_ett>>12) & 0x1 ;

  int psb_htt = psb2.bData(4);
  int htt_rank = psb_htt & 0xfff;
  gt_.RankHTT = htt_rank;
  gt_.OvHTT   = (psb_htt>>12) & 0x1 ;

  int psb_etmis = psb2.aData(5);
  int etmis_rank = psb_etmis & 0xfff;
  int psb_etmis_phi = psb2.bData(5) & 0x7F ;
  gt_.RankETM = etmis_rank;
  gt_.PhiETM  = psb_etmis_phi;
  gt_.OvETM   = (psb_etmis>>12) & 0x1 ;

  int psb_htmis = psb2.aData(3);
  int htmis_rank = (psb_htmis >> 5) & 0x7f;
  int htmis_phi = psb_htmis & 0x1F ;
  gt_.RankHTM = htmis_rank;
  gt_.PhiHTM  = htmis_phi;
  gt_.OvHTM   = (psb_htmis >> 12) & 0x1 ;

}

// =---------------------------------------------

      std::vector<int> psbel;
      psbel.push_back(psb.aData(4));
      psbel.push_back(psb.aData(5));
      psbel.push_back(psb.bData(4));
      psbel.push_back(psb.bData(5));
      std::vector<int>::const_iterator ipsbel;
      for(ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
        float rank = (*ipsbel)&0x3f;
        if(rank>0) {
          gt_.Bxel.push_back(ibx);
          gt_.Rankel.push_back(rank);
          gt_.Phiel.push_back(((*ipsbel)>>10)&0x1f);
          gt_.Etael.push_back(( ((*ipsbel>>9)&1) ? 10-(((*ipsbel)>>6)&7) : (((*ipsbel)>>6)&7)+11 )); 
	  gt_.Isoel.push_back(false);
       }
      }
      psbel.clear();
      psbel.push_back(psb.aData(6));
      psbel.push_back(psb.aData(7));
      psbel.push_back(psb.bData(6));
      psbel.push_back(psb.bData(7));
      for(ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
        float rank = (*ipsbel)&0x3f;
        if(rank>0) {
          gt_.Bxel.push_back(ibx);
          gt_.Rankel.push_back(rank);
          gt_.Phiel.push_back(((*ipsbel)>>10)&0x1f);
          gt_.Etael.push_back(( ((*ipsbel>>9)&1) ? 10-(((*ipsbel)>>6)&7) : (((*ipsbel)>>6)&7)+11 )); 
	  gt_.Isoel.push_back(true);
        }
      }


      // central jets
      std::vector<int> psbjet;
      psbjet.push_back(psb.aData(2));
      psbjet.push_back(psb.aData(3));
      psbjet.push_back(psb.bData(2));
      psbjet.push_back(psb.bData(3));
      std::vector<int>::const_iterator ipsbjet;
      for(ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
        float rank = (*ipsbjet)&0x3f;
        if(rank>0) {
          gt_.Bxjet.push_back(ibx);
          gt_.Rankjet.push_back(rank);
          gt_.Phijet.push_back(((*ipsbjet)>>10)&0x1f);
          gt_.Etajet.push_back(( ((*ipsbjet>>9)&1) ? 10-(((*ipsbjet)>>6)&7) : (((*ipsbjet)>>6)&7)+11 ));
	  gt_.Taujet.push_back(false);
	  gt_.Fwdjet.push_back(false);
         }
      }

      // tau jets
      psbjet.clear();
      psbjet.push_back(psb2.aData(6));
      psbjet.push_back(psb2.aData(7));
      psbjet.push_back(psb2.bData(6));
      psbjet.push_back(psb2.bData(7));
      for(ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
        float rank = (*ipsbjet)&0x3f;
        if(rank>0) {
          gt_.Bxjet.push_back(ibx);
          gt_.Rankjet.push_back(rank);
          gt_.Phijet.push_back(((*ipsbjet)>>10)&0x1f);
          gt_.Etajet.push_back(( ((*ipsbjet>>9)&1) ? 10-(((*ipsbjet)>>6)&7) : (((*ipsbjet)>>6)&7)+11 ));
	  gt_.Taujet.push_back(true);
	  gt_.Fwdjet.push_back(false);          
        }
      }


      // forward jets
      psbjet.clear();
      psbjet.push_back(psb.aData(0));
      psbjet.push_back(psb.aData(1));
      psbjet.push_back(psb.bData(0));
      psbjet.push_back(psb.bData(1));
      for(ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
        float rank = (*ipsbjet)&0x3f;
        if(rank>0) {
          gt_.Bxjet.push_back(ibx);
          gt_.Rankjet.push_back(rank);
          gt_.Phijet.push_back(((*ipsbjet)>>10)&0x1f);
          gt_.Etajet.push_back(( ((*ipsbjet>>9)&1) ? 3-(((*ipsbjet)>>6)&7) : (((*ipsbjet)>>6)&7)+18 ));
	  gt_.Taujet.push_back(false);
	  gt_.Fwdjet.push_back(true);         
        }
      }

    }
    gt_.Nele = gt_.Bxel.size();
    gt_.Njet = gt_.Bxjet.size();


    L1GtFdlWord fdlWord = gtrr->gtFdlWord();
    
    
    /// get Global Trigger algo and technical triger bit statistics
    gt_.tw1.resize(5,0);
    gt_.tw2.resize(5,0);
    gt_.tt.resize(5,0);
    
    for(int iebx=0; iebx<5; iebx++)
    {
      DecisionWord gtDecisionWord = gtrr->decisionWord(iebx-2);

      int dbitNumber = 0;

      DecisionWord::const_iterator GTdbitItr;
      for(GTdbitItr = gtDecisionWord.begin(); GTdbitItr != gtDecisionWord.end(); GTdbitItr++) {
        if (*GTdbitItr) {
          if(dbitNumber<64) { gt_.tw1[iebx] |= (1LL<<dbitNumber); }
          else { gt_.tw2[iebx] |= (1LL<<(dbitNumber-64)); }
        }
        dbitNumber++; 
      }

      dbitNumber = 0;
      TechnicalTriggerWord gtTTWord = gtrr->technicalTriggerWord(iebx-2);
      TechnicalTriggerWord::const_iterator GTtbitItr;
      for(GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
        if (*GTtbitItr) {
          gt_.tt[iebx] |= (1LL<<dbitNumber);
        }
        dbitNumber++;
      }
    }
}



