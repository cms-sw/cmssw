#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexUpdator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

KinematicConstrainedVertexUpdator::KinematicConstrainedVertexUpdator()
{ 
 vFactory  = new KinematicVertexFactory();
 vConstraint = new VertexKinematicConstraint();
}
  
KinematicConstrainedVertexUpdator::~KinematicConstrainedVertexUpdator()
{
 delete vFactory;
 delete vConstraint;
}

std::pair<std::pair<std::vector<KinematicState>, AlgebraicMatrix >, RefCountedKinematicVertex >
KinematicConstrainedVertexUpdator::update(const AlgebraicVector& inPar,
	const AlgebraicMatrix& inCov, const std::vector<KinematicState> &lStates,
	const GlobalPoint& lPoint, MultiTrackKinematicConstraint * cs)const
{
 const MagneticField* field=lStates.front().magneticField();
 AlgebraicMatrix d_matrix = vConstraint->parametersDerivative(lStates, lPoint);
 AlgebraicMatrix e_matrix = vConstraint->positionDerivative(lStates, lPoint);
 AlgebraicVector val_s = vConstraint->value(lStates, lPoint);
 int vSize = lStates.size();

//delta alpha
 AlgebraicVector d_a(7*vSize + 3);
 d_a(1) = lPoint.x();
 d_a(2) = lPoint.y();
 d_a(3) = lPoint.z();
 
 int cst = 0;
 for(std::vector<KinematicState>::const_iterator i = lStates.begin(); i != lStates.end(); i++)
 {
  AlgebraicVector lst_par = asHepVector<7>(i->kinematicParameters().vector());
  for(int j = 1; j<lst_par.num_row()+1; j++)
  {d_a(3+7*cst+j) = lst_par(j);}
  cst++;
 }
 
 
 AlgebraicVector delta_alpha = inPar - d_a; 

// cout<<"delta_alpha"<<delta_alpha<<endl;
//resulting matrix of derivatives and vector of values.
//their size  depends of number of tracks to analyze and number of
//additional constraints to apply 
  AlgebraicMatrix g;
  AlgebraicVector val;
  if(cs == 0)
  {
 
//unconstrained vertex fitter case
   g = AlgebraicMatrix(2*vSize,7*vSize+3,0);
   val = AlgebraicVector(2*vSize);
  
//filling the derivative matrix  
   g.sub(1,1,e_matrix);
   g.sub(1,4,d_matrix);
  
//filling the vector of values  
   val = val_s;
  }else{
 
//constrained vertex fitter case 
   int n_eq =cs->numberOfEquations();
   g = AlgebraicMatrix(n_eq + 2*vSize,7*vSize + 3,0);
   val = AlgebraicVector(n_eq + 2*vSize);
   AlgebraicMatrix n_x = cs->positionDerivative(lStates, lPoint); 
   AlgebraicMatrix n_alpha = cs->parametersDerivative(lStates, lPoint);
   AlgebraicVector c_val = cs->value(lStates, lPoint);
  
//filling the derivative matrix  
//   cout<<"n_x:"<<n_x<<endl;
//   cout<<"n_alpha"<<n_alpha<<endl;

   g.sub(1,1,n_x);
   g.sub(1,4,n_alpha);
   g.sub(n_eq+1, 1, e_matrix);
   g.sub(n_eq+1, 4, d_matrix);

//filling the vector of values  
   for(int i = 1;i< n_eq+1; i++)
   {val(i) = c_val(i);}
   for(int i = 1; i<(2*vSize+1); i++)
   {val(i+n_eq) = val_s(i);} 
  }

  //check for NaN
  for(int i = 1; i<=val.num_row();++i) {
    if (edm::isNotFinite(val(i))) {
      LogDebug("KinematicConstrainedVertexUpdator")
      << "catched NaN.\n";
      return std::pair<std::pair<std::vector<KinematicState>, AlgebraicMatrix>, RefCountedKinematicVertex >(
        std::pair<std::vector<KinematicState>, AlgebraicMatrix>(std::vector<KinematicState>(), AlgebraicMatrix(1,0)),
        RefCountedKinematicVertex());   
    }
  }

//debug feature  
  AlgebraicSymMatrix in_cov_sym(7*vSize + 3,0);

  for(int i = 1; i<7*vSize+4; ++i)
  {
   for(int j = 1; j<7*vSize+4; ++j)
   {if(i<=j) in_cov_sym(i,j) = inCov(i,j);}
  }  
    
//debug code   
  AlgebraicSymMatrix v_g_sym = in_cov_sym.similarity(g);

  int ifl1 = 0;
  v_g_sym.invert(ifl1);
  if(ifl1 !=0) {
    LogDebug("KinematicConstrainedVertexFitter")
	<< "Fit failed: unable to invert SYM gain matrix\n";
    return std::pair<std::pair<std::vector<KinematicState>, AlgebraicMatrix>, RefCountedKinematicVertex >(
	std::pair<std::vector<KinematicState>, AlgebraicMatrix>(std::vector<KinematicState>(), AlgebraicMatrix(1,0)),
	RefCountedKinematicVertex());	
  }
 
// delta alpha is now valid!
//full math case now!
  AlgebraicVector lambda = v_g_sym *(g*delta_alpha + val);
  
//final parameters  
  AlgebraicVector finPar = inPar -  in_cov_sym * g.T() * lambda;

//covariance matrix business:
  AlgebraicMatrix mFactor = in_cov_sym *(v_g_sym.similarityT(g))* in_cov_sym;
  
//refitted covariance 
  AlgebraicMatrix rCov = in_cov_sym - mFactor;
   
//symmetric covariance:    
  AlgebraicSymMatrix r_cov_sym(7*vSize+3,0);  
  for(int i = 1; i<7*vSize+4; ++i)
  {
   for(int j = 1; j<7*vSize+4; ++j)
   {if(i<=j)r_cov_sym(i,j)  = rCov(i,j);}
  }
    
  AlgebraicSymMatrix pCov = r_cov_sym.sub(1,3);

// chi2
  AlgebraicVector chi  = lambda.T()*(g*delta_alpha  + val);
 
//this is ndf without significant prior
//vertex so -3 factor exists here 
  float ndf = 2*vSize - 3;
  if(cs != 0){ndf += cs->numberOfEquations();}
 

//making resulting vertex 
  GlobalPoint vPos (finPar(1),finPar(2),finPar(3));
  VertexState st(vPos,GlobalError( asSMatrix<3>(pCov)));
  RefCountedKinematicVertex rVtx = vFactory->vertex(st,chi(1),ndf);

//making refitted states of Kinematic Particles
  int i_int = 0;
  std::vector<KinematicState> ns;
  for(std::vector<KinematicState>::const_iterator i_st=lStates.begin(); i_st != lStates.end(); i_st++)
  {
   AlgebraicVector7 newPar; 
   for(int i =0; i<7; i++)
   {newPar(i) = finPar(4 + i_int*7 + i);}
  
   AlgebraicSymMatrix nCovariance = r_cov_sym.sub(4 + i_int*7, 10 + i_int*7);
   TrackCharge chl = i_st->particleCharge();
   KinematicParameters nrPar(newPar);
   KinematicParametersError nrEr(asSMatrix<7>(nCovariance));
   KinematicState newState(nrPar,nrEr,chl, field);
   ns.push_back(newState);
   i_int++;
  }
 std::pair<std::vector<KinematicState>, AlgebraicMatrix> ns_m(ns,rCov);
 return std::pair<std::pair<std::vector<KinematicState>, AlgebraicMatrix>, RefCountedKinematicVertex >(ns_m,rVtx);	
}
