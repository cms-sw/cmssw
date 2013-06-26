#ifndef VertexMass_H  
#define VertexMass_H 

class TransientVertex;

class VertexMass {

public:

  VertexMass(); 

  VertexMass(double pionMass); 

  ~VertexMass() {};

  double operator()(const TransientVertex &) const;
  
private:

  double thePionMass; 
  
};

#endif

