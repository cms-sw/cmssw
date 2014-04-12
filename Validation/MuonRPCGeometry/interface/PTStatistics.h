#ifndef RPCPatts_PTStatistics_h
#define RPCPatts_PTStatistics_h


#include <vector>
#include <string>
#include <algorithm>


#include "Validation/MuonRPCGeometry/interface/Constants.h"

class PTStatistics: public std::vector<long long> {
  public:
    PTStatistics();
    void update(PTStatistics & otherPtStats);
    long int sum(const int & ptCut) const;
    long int sum() const;
    long double sumR() const;     
    long double sumR(const int & ptCut) const;

    
    long double eff(int ptCut);
    std::string toString();
    
    static const std::vector<long double> m_rates; // used for pur calculation
    
};


#endif
