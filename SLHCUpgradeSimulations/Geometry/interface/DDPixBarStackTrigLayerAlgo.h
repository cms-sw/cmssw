#ifndef DD_PixBarStackTrigLayerAlgo_h
#define DD_PixBarStackTrigLayerAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixBarStackTrigLayerAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarStackTrigLayerAlgo(); 
  virtual ~DDPixBarStackTrigLayerAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              genMat;      //Name of general material
  int                      number;      //Number of ladders in phi
  double                   layerDz;     //Length of the layer
  double                   layerR;      //Radius of the layer
  std::vector<std::string> ladder;      //Names     of ladders
  std::vector<double>      ladderWidth; //Widths         ...
  std::vector<double>      ladderThick; //Thickness      ...
};

#endif
