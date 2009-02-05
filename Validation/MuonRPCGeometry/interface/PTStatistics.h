#ifndef RPCPatts_PTStatistics_h
#define RPCPatts_PTStatistics_h


#include <vector>
#include <string>
#include <algorithm>


#include "Validation/MuonRPCGeometry/interface/Constants.h"

class PTStatistics: public std::vector<int> {
  public:
    PTStatistics();
    void update(PTStatistics & otherPtStats);
    int sum(const int & ptCut) const;
    int sum() const;
    double sumR() const;     
    double sumR(const int & ptCut) const;

    
    double eff(int ptCut);
    std::string toString();
    
    static bool rateInitilized;
    static std::vector<double> m_rates; // used for pur calculation
    
};


#endif
