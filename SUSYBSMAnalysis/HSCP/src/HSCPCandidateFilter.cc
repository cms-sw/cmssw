#include "SUSYBSMAnalysis/HSCP/interface/HSCPCandidateFilter.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

// utility public methods.
void HSCPCandidateFilter::trim(std::vector<HSCParticle>& candidates, cutName cut) const {
  std::vector<HSCParticle>::iterator it = candidates.begin();
  while(it<candidates.end()) {
    if(passCut(*it,cut)) {
      ++it;
    } else {
      it = candidates.erase(it);
    }
  }
}

std::vector<HSCParticle> HSCPCandidateFilter::filter(std::vector<HSCParticle>& candidates, cutName cut) const {
  std::vector<HSCParticle> result;
  for(std::vector<HSCParticle>::const_iterator it = candidates.begin(); it<candidates.end(); ++it)
    if(passCut(*it,cut)) result.push_back(*it);
  return result;
}

std::string HSCPCandidateFilter::nameCut(cutName cut) const {
  switch(cut) {
    case 0:
      return "All";
    case 1:
      return "NoSel";
    case 2:
      return "DT-090";
    case 3:
      return "TK-085";
    case 4:
      return "DT-080";
    case 5:
      return "TK-080";
    case 6:
      return "ME-20MA";
    case 7:
      return "DB-01";
    case 8:
      return "MM-140";
    case 9:
      return "DTE-07";
    case 10:
      return "DTE-10";
    case 11:
      return "DT-085_DTE-10";
    case 12:
      return "DT-080_DTE-10";
    case 13:
      return "TK-080_DT-080";
    case 14:
      return "TK-080_DT-085_DTE-10";
    case 15:
      return "TK-080_DT-080_DTE-10";
    case 16:
      return "TK-080_DT-080_DTE-10_DB-01";
    case 17:
      return "TK-080_DT-080_DTE-07";
    case 18:
      return "TK-080_DT-080_ME-15";
    case 19:
      return "TK-080_DT-080_ME-15_MM-100";
    case 20:
      return "TK-080_DT-080_ME-15_MM-200";
    case 21:
      return "TK-080_DT-080_ME-15_MM-300";
    case 22:
      return "TK-080_DT-080_ME-15_MM-600";
    case 23:
      return "TK-080_DB-01_DT-090_MM-140_TKM-100_DTM-100";
    case 24:
      return "TK-080_TKPT-100";
    case 25:
      return "TK-080_TKPT-100_DEDH-14";
    case 26:
      return "TK-080_TKPT-100_TKM-100";
    case 27:
      return "TK-080_TKPT-100_TKM-200";
    case 28:
      return "TK-080_TKPT-100_TKM-400";
    case 29: 
      return "CandidateDeDx";
    case 30:
      return "CandidateTOF";
    case 31:
      return "Candidate";
    default:
      return "";
  }
  return "";
}

bool HSCPCandidateFilter::passCut(const HSCParticle& candidate, cutName cut) const {
  switch(cut) {
    case 0:
      return true;
    case 1:
      return nosel(candidate);
    case 2:
      return passCut(candidate,NoSel)&&dt090(candidate);
    case 3:
      return passCut(candidate,NoSel)&&dt085(candidate);
    case 4:
      return passCut(candidate,NoSel)&&dt080(candidate);
    case 5:
      return passCut(candidate,NoSel)&&tk080(candidate);
    case 6:
      return passCut(candidate,NoSel)&&me20ma(candidate);
    case 7:
      return passCut(candidate,NoSel)&&db01(candidate);
    case 8:
      return passCut(candidate,NoSel)&&mm140(candidate);
    case 9:
      return passCut(candidate,NoSel)&&dte07(candidate);
    case 10:
      return passCut(candidate,NoSel)&&dte10(candidate);
    case 11:
      return passCut(candidate,TK085)&&passCut(candidate,DTE10);
    case 12:
      return passCut(candidate,DT080)&&passCut(candidate,DTE10);
    case 13:
      return passCut(candidate,DT080)&&passCut(candidate,TK080);
    case 14:
      return passCut(candidate,TK080)&&passCut(candidate,DT085_DTE10);
    case 15:
      return passCut(candidate,TK080)&&passCut(candidate,DT080_DTE10);
    case 16:
      return passCut(candidate,TK080_DT080_DTE10)&&passCut(candidate,DB01);
    case 17:
      return passCut(candidate,TK080_DT080_DTE10)&&passCut(candidate,DTE07);
    case 18:
      return passCut(candidate,TK080_DT080)&&me15(candidate);
    case 19:
      return passCut(candidate,TK080_DT080_ME15)&&mm100(candidate);
    case 20:
      return passCut(candidate,TK080_DT080_ME15)&&mm200(candidate);
    case 21:
      return passCut(candidate,TK080_DT080_ME15)&&mm300(candidate);
    case 22:
      return passCut(candidate,TK080_DT080_ME15)&&mm600(candidate);
    case 23:
      return passCut(candidate,TK080)&&passCut(candidate,DB01)&&passCut(candidate,MM140)&&passCut(candidate,DT090)
             &&tkm100(candidate)&&dtm100(candidate);
    case 24:
      return passCut(candidate,TK080)&&tkpt100(candidate);
    case 25:
      return passCut(candidate,TK080_TKPT100)&&tkhits14(candidate);
    case 26:
      return passCut(candidate,TK080_TKPT100)&&tkm100(candidate);
    case 27:
      return passCut(candidate,TK080_TKPT100)&&tkm200(candidate);
    case 28:
      return passCut(candidate,TK080_TKPT100)&&tkm400(candidate);
    case 29: 
      return tkm100(candidate);
    case 30:
      return dtm100(candidate);
    case 31:
      return passCut(candidate,NoSel) && passCut(candidate,CandidateDeDx) && passCut(candidate,CandidateTOF);
    default:
      return false;
  }
  return false;
}

// actual methods to apply cuts. 
// They are kept separated to ease documentation. Name can be used too for that purpose.
bool HSCPCandidateFilter::nosel(const HSCParticle& candidate) const {
  return (candidate.hasTkInfo() && candidate.hasDtInfo() && candidate.hasMuonTrack());
}

bool HSCPCandidateFilter::dte07(const HSCParticle& candidate) const {
  return (candidate.Dt().second.invBetaErr < 0.07); 
}

bool HSCPCandidateFilter::dte10(const HSCParticle& candidate) const {
  return (candidate.Dt().second.invBetaErr < 0.10); 
}

bool HSCPCandidateFilter::dt080(const HSCParticle& candidate) const {
  return (candidate.Dt().second.invBeta > 1.25 && candidate.Dt().second.invBeta < 1000.);
}

bool HSCPCandidateFilter::dt085(const HSCParticle& candidate) const {
  return (candidate.Dt().second.invBeta > 1.176 &&  candidate.Dt().second.invBeta < 1000.);
}

bool HSCPCandidateFilter::dt090(const HSCParticle& candidate) const {
  return (candidate.Dt().second.invBeta > 1.11);
}

bool HSCPCandidateFilter::tk080(const HSCParticle& candidate) const {
  return (candidate.Tk().invBeta2() > 1.56);
}

bool HSCPCandidateFilter::tkpt100(const HSCParticle& candidate) const {
  return (candidate.Tk().track()->pt() > 100 &&
          candidate.Dt().first->combinedMuon().isNonnull() && candidate.Dt().first->combinedMuon()->pt() > 100);
}

bool HSCPCandidateFilter::stapt100(const HSCParticle& candidate) const {
  return (candidate.Dt().first->standAloneMuon()->pt() > 100);
}

bool HSCPCandidateFilter::tkm100(const HSCParticle& candidate) const {
  return (candidate.massTk() > 100);
}

bool HSCPCandidateFilter::tkm200(const HSCParticle& candidate) const {
  return (candidate.massTk() > 200);
}

bool HSCPCandidateFilter::tkm400(const HSCParticle& candidate) const {
  return (candidate.massTk() > 400);
}

bool HSCPCandidateFilter::dtm100(const HSCParticle& candidate) const {
  return (candidate.massDt() > 100);
}

bool HSCPCandidateFilter::dtm200(const HSCParticle& candidate) const {
  return (candidate.massDt() > 200);
}

bool HSCPCandidateFilter::dtm400(const HSCParticle& candidate) const {
  return (candidate.massDt() > 400);
}

bool HSCPCandidateFilter::tkhits14(const HSCParticle& candidate) const {
  return (candidate.Tk().nDedxHits() >= 14); 
}

bool HSCPCandidateFilter::db01(const HSCParticle& candidate) const {
  return (fabs(sqrt(1./candidate.Tk().invBeta2())- 1./candidate.Dt().second.invBeta ) < 0.1);
}

bool HSCPCandidateFilter::me20ma(const HSCParticle& candidate) const {
  return (candidate.massAvgError() < 0.05 + 0.2*candidate.massAvg()/1000.);
}

bool HSCPCandidateFilter::me15(const HSCParticle& candidate) const {
  return candidate.massAvgError() < 0.15 ;
}
	
bool HSCPCandidateFilter::mm100(const HSCParticle& candidate) const {
  return (candidate.massAvg() > 100);
}

bool HSCPCandidateFilter::mm140(const HSCParticle& candidate) const {
  return (candidate.massAvg() > 140);
}

bool HSCPCandidateFilter::mm200(const HSCParticle& candidate) const {
  return (candidate.massAvg() > 200);
}

bool HSCPCandidateFilter::mm300(const HSCParticle& candidate) const {
  return (candidate.massAvg() > 300);
}

bool HSCPCandidateFilter::mm600(const HSCParticle& candidate) const {
  return (candidate.massAvg() > 600);
}

}

