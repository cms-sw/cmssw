
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef INFO_HELPER_H
#define INFO_HELPER_H

// system include files
#include <map>
#include <vector>
#include <string>

//void LocalStubBuilder<T>::beginJob(const edm::EventSetup&) [with T = cmsUpgrades::Ref_PSimHit_]
//cmsUpgrades::ClusteringAlgorithm_a<T>::ClusteringAlgorithm_a(const cmsUpgrades::StackedTrackerGeometry*) [with T = cmsUpgrades::Ref_PSimHit_]
//cmsUpgrades::HitMatchingAlgorithm_globalgeometry<T>::HitMatchingAlgorithm_globalgeometry(const cmsUpgrades::StackedTrackerGeometry*, double, double) [with T = cmsUpgrades::Ref_PSimHit_]
//void LocalStubBuilder<T>::beginJob(const edm::EventSetup&) [with T = cmsUpgrades::Ref_PixelDigi_]
//cmsUpgrades::ClusteringAlgorithm_a<T>::ClusteringAlgorithm_a(const cmsUpgrades::StackedTrackerGeometry*) [with T = cmsUpgrades::Ref_PixelDigi_]
//cmsUpgrades::HitMatchingAlgorithm_globalgeometry<T>::HitMatchingAlgorithm_globalgeometry(const cmsUpgrades::StackedTrackerGeometry*, double, double) [with T = cmsUpgrades::Ref_PixelDigi_]

class classInfo {
 
  public:
  classInfo( std::string aPrettyFunction )
  {
      std::string::size_type pos[10];
 //----------------------------------------------------------------------------------------------------------
 // Parse the single string into three sub-strings:
 // 1) the return-type, the class name and the function name
 // 2) the function arguments
 // 3) the template expressions
   std::string returntype_class_func, args, templates;
 // Find the delimiter tokens
   pos[0] = aPrettyFunction.find("(");
   pos[1] = aPrettyFunction.find(")");
   pos[2] = aPrettyFunction.find("[");
   pos[3] = aPrettyFunction.find("]");
 // assign the return-type, class name and function name string
   returntype_class_func = aPrettyFunction.substr(0,pos[0]);
 // assign the arguments string
   args = aPrettyFunction.substr(pos[0]+1,pos[1]-pos[0]-1);
 // assign the template expressions string (if there are any template parameters)
   if( (std::string::npos != pos[2])&&(std::string::npos != pos[3]) )
    templates = aPrettyFunction.substr(pos[2]+6,pos[3]-pos[2]-7);
 //----------------------------------------------------------------------------------------------------------
 
 
 
 //----------------------------------------------------------------------------------------------------------
 // Parse the return-type, class name, function name string into components and store in member variables
   if( (pos[4]=returntype_class_func.find(" ")) != std::string::npos ){
    returnType = returntype_class_func.substr(0,pos[4]);
    returntype_class_func = returntype_class_func.substr(pos[4]);
   }
   if( (pos[4]=returntype_class_func.rfind("::")) != std::string::npos ){
    functionName = returntype_class_func.substr(pos[4]+2);
    returntype_class_func = returntype_class_func.substr(0,pos[4]);
   }else{
    functionName = returntype_class_func;
    returntype_class_func = "";
   }
   if( (pos[4]=returntype_class_func.rfind("::")) != std::string::npos ){
    className = returntype_class_func.substr(pos[4]+2);
    nameSpaces = returntype_class_func.substr(0,pos[4]);
   }else{
    className = returntype_class_func;
    returntype_class_func = "";
   }
 //----------------------------------------------------------------------------------------------------------
 
 
 
 //----------------------------------------------------------------------------------------------------------
 // Parse the function arguments in a far more sensible way than previously
   int bracket_count = 0;
   size_t last = 0;
   for(unsigned int i = 0; i != args.size() ; ++i ){
    if ( args[i] == '<' ) bracket_count++;
    else if ( args[i] == '>' ) bracket_count--;
    else if ( bracket_count==0 ){
     if ( args[i] == ',' ){
      functionArguments.push_back( args.substr( last , i-last ) );
      last = i;
     }
    }
   }
   functionArguments.push_back( args.substr( last ) );
 //----------------------------------------------------------------------------------------------------------
 
 
 
 //----------------------------------------------------------------------------------------------------------
 // Parse the template parameter definitions in a far more sensible way than previously
 // First into separate string...
   std::vector<std::string> templateStrings;
   bracket_count = 0;
   last = 0;
   for(unsigned int i = 0; i != templates.size() ; ++i ){
    if ( templates[i] == '<' ) bracket_count++;
    else if ( templates[i] == '>' ) bracket_count--;
    else if ( bracket_count==0 ){
     if ( templates[i] == ',' ){
      templateStrings.push_back( templates.substr( last , i-last ) );
      last = i;
     }
    }
   }
   templateStrings.push_back( templates.substr( last ) );
 
 // ...then for each string, split them and store them in the map
   for(std::vector<std::string>::const_iterator it=templateStrings.begin();it!=templateStrings.end();++it){
    pos[5]=it->find("=");
    templateTypes.insert( std::make_pair( it->substr(0,pos[5]-1), it->substr(pos[5]+2) ) );
   }
 //----------------------------------------------------------------------------------------------------------
  }
  ~classInfo(){}
 
  const std::string &ReturnType() const {return returnType;}
  const std::string &NameSpaces() const {return nameSpaces;}
  const std::string &ClassName() const {return className;}
  const std::string &FunctionName() const {return functionName;}
  const std::vector<std::string> &FunctionArguments() const {return functionArguments;}
  const std::map< std::string , std::string > &TemplateTypes() const {return templateTypes;}
 
  private:
  std::string returnType;
  std::string nameSpaces;
  std::string className;
  std::string functionName;
  std::vector<std::string> functionArguments;
  std::map< std::string , std::string > templateTypes;
 };
 

#endif
