#ifndef V0SvFilter_H  
#define V0SvFilter_H 

class TransientVertex;

// provides a simple filter to reject vertices compatible
// with a K0s hypothesis (two tracks of opposite charge
// and an invariant mass within a adjustable window around 
// the nominal k0s mass

class V0SvFilter {

public:

  V0SvFilter(double massWindow); 

  ~V0SvFilter() {};

  bool operator()(const TransientVertex &) const;
  
private:

  double theMassWindow; 

  double theK0sMass;
  
};

#endif

