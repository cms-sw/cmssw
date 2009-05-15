#ifndef SUSYBSMANALYSIS_HSCPCandidateFilter_HSCPCandidateFilter_H
#define SUSYBSMANALYSIS_HSCPCandidateFilter_HSCPCandidateFilter_H

#include <vector>
#include <string>

namespace susybsm {

class HSCParticle;

class HSCPCandidateFilter
{
  public:
    enum cutName { All=0,
                   NoSel,DT090,TK085,DT080,TK080,ME20MA,DB01,MM140,DTE07,
		   DTE10,DT085_DTE10,DT080_DTE10,TK080_DT080,TK080_DT085_DTE10,
		   TK080_DT080_DTE10,TK080_DT080_DTE10_DB01,TK080_DT080_DTE07,
		   TK080_DT080_ME15,TK080_DT080_ME15_MM100,TK080_DT080_ME15_MM200,
		   TK080_DT080_ME15_MM300,TK080_DT080_ME15_MM600,
		   TK080_DB01_DT090_MM140_TKM100_DTM100,TK080_TKPT100,
		   TK080_TKPT100_DEDH14,TK080_TKPT100_M100,TK080_TKPT100_M200,
		   TK080_TKPT100_M400,CandidateDeDx,CandidateTOF,Candidate 
		 };
  
    HSCPCandidateFilter() {}
    virtual ~HSCPCandidateFilter() {}
    void trim(std::vector<HSCParticle>& candidates, cutName cut) const;
    std::vector<HSCParticle> filter(std::vector<HSCParticle>& candidates, cutName cut) const;
    bool passCut(const HSCParticle& candidate, cutName cut) const;
    std::string nameCut(cutName cut) const;
  private:
    bool nosel(const HSCParticle&) const;
    bool dte07(const HSCParticle&) const;
    bool dte10(const HSCParticle&) const;
    bool dt080(const HSCParticle&) const;
    bool dt085(const HSCParticle&) const;
    bool dt090(const HSCParticle&) const;
    bool tk080(const HSCParticle&) const;
    bool tkpt100(const HSCParticle&) const;
    bool stapt100(const HSCParticle&) const;
    bool tkm100(const HSCParticle&) const;
    bool tkm200(const HSCParticle&) const;
    bool tkm400(const HSCParticle&) const;
    bool dtm100(const HSCParticle&) const;
    bool dtm200(const HSCParticle&) const;
    bool dtm400(const HSCParticle&) const;
    bool tkhits14(const HSCParticle&) const;
    bool db01(const HSCParticle&) const;
    bool me20ma(const HSCParticle&) const;
    bool me15(const HSCParticle&) const;
    bool mm100(const HSCParticle&) const;
    bool mm140(const HSCParticle&) const;
    bool mm200(const HSCParticle&) const;
    bool mm300(const HSCParticle&) const;
    bool mm600(const HSCParticle&) const;
};

}

#endif
