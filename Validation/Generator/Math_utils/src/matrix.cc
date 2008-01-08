/**********************************
Implementation of a simple matrix class
Bruce Knuteson 2003
**********************************/


#include "VistaTools/Math_utils/interface/matrix.hh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif

/*****  Constructors  *****/

matrix::matrix(size_type n, size_type m): data(n) 
{
  for(size_type i=0; i<n; i++)
    data[i] = vector<double>(m,0.);
}

matrix::matrix(const data_t & _data): data(_data)
{
}

/*****   Accessors   *****/

matrix::size_type matrix::nrows() const
{
  return data.size();
}
	
matrix::size_type matrix::ncols() const
{
  if(data.empty())
    return 0;
  return data[0].size();
}

vector<double> & matrix::operator[](matrix::size_type row) { return data[row]; }

const vector<double> & matrix::operator[](matrix::size_type row) const { return data[row]; }
	


/*****   Modifiers   *****/

void matrix::resize(matrix::size_type n, matrix::size_type m)
{
  if(n<0) n=0;
  if(m<0) m=0;
  matrix::size_type n0 = nrows();
  matrix::size_type m0 = ncols();
  if(n0>n)
    for(matrix::size_type j=n0-1; j>=n; j--)
      data.pop_back();
  if(n0<n)
    for(matrix::size_type j=n0; j<n; j++)
      data.push_back(vector<double>(m0)); 
  //data.resize(n);
  for(matrix::size_type i=0; i<n; i++)
    {
      if(m0>m)
	for(matrix::size_type j=m0-1; j>=m; j--)
	  data[i].pop_back();
      if(m0<m)
	for(matrix::size_type j=m0; j<m; j++)
	  data[i].push_back(0.);
      //data[i].resize(m);
    }
}
	
matrix & matrix::operator=(matrix rhs)
{
  this->resize(rhs.nrows(), rhs.ncols());
  for(matrix::size_type i=0; i<nrows(); i++) data[i] = rhs.data[i];
  return *this;
}

void matrix::deletecolumn(matrix::size_type n)
{
  for(matrix::size_type i=0; i<this->nrows(); i++)
    for(matrix::size_type j=n; j<this->ncols()-1; j++)
      data[i][j] = data[i][j+1];
  this->resize(this->nrows(),this->ncols()-1);
}

void matrix::deleterow(matrix::size_type n)
{
  for(matrix::size_type i=n; i<this->nrows()-1; i++)
    this->data[i]=this->data[i+1];
  this->resize(this->nrows()-1,this->ncols());
}


/*****    Methods    *****/

void matrix::print(std::ostream & fout) 
{
  for(matrix::size_type i=0; i<this->nrows(); i++)
    {
      for(matrix::size_type j=0; j<this->ncols(); j++) fout << this->data[i][j] << "\t";
      fout << endl;
    }
}

vector<double> matrix::getEigenvalues()
{
  matrix temp = *this;
  assert(nrows()==ncols()); // works only when nrows() == ncols()
  matrix::size_type nvars = nrows(); 
  vector<double> ansEigenvalues(nvars);
  matrix ansEigenvectors(nvars,nvars);
  jacobi(temp, nvars, ansEigenvalues, ansEigenvectors);
  return(ansEigenvalues);
}

matrix matrix::power(double n) const
{
  if(n==1)
    return(*this);
  matrix ans;
  assert(nrows()==ncols()); // inversion works only when nrows() == ncols()
  matrix::size_type nvars = this->nrows(); 
  ans.resize(nvars,nvars);
  matrix temp(nvars, nvars);
  for(matrix::size_type j=0; j<nvars; j++)
    for(matrix::size_type k=0; k<nvars; k++)
      temp[j][k] = this->data[j][k];
  vector<double> ansEigenvalues(nvars);
  matrix ansEigenvectors(nvars,nvars);
  jacobi(temp, nvars, ansEigenvalues, ansEigenvectors);
  for(matrix::size_type j=0; j<nvars; j++)
    for(matrix::size_type k=0; k<nvars; k++)
      {
	ans[j][k]=0.;
	for(matrix::size_type l=0; l<nvars; l++)
	  {
	    ans[j][k]+=
	      ansEigenvectors[j][l]*ansEigenvectors[k][l]*
	      pow(ansEigenvalues[l],n);
	  }
      }
  return ans;
}

double matrix::det()
{
  matrix::size_type nvars = this->nrows(); // inversion works only when nrows() == ncols()
  matrix temp(nvars, nvars);
  for(matrix::size_type j=0; j<nvars; j++)
    for(matrix::size_type k=0; k<nvars; k++)
      temp[j][k] = this->data[j][k];
  vector<double> ansEigenvalues(nvars);
  matrix ansEigenvectors(nvars,nvars);
  jacobi(temp, nvars, ansEigenvalues, ansEigenvectors);
  double ans = 1.;
  for(matrix::size_type l=0; l<nvars; l++)
    ans *= ansEigenvalues[l];
  return ans;
}  

matrix matrix::safeInverse(double & determinant)
{
  matrix temp = *this;
  assert(this->ncols()==this->nrows());
  if(ncols()==0)
    {
      determinant = 1.;
      return(*this);
    }
  matrix::size_type nvars = this->ncols();
  vector<double> ansEigenvalues(nvars);
  matrix ansEigenvectors(nvars,nvars);
  jacobi(temp, nvars, ansEigenvalues, ansEigenvectors);

  double maxEigenvalue = 1.;
  for(matrix::size_type j=0; j<nvars; j++)
    if(ansEigenvalues[j]>maxEigenvalue)
      maxEigenvalue = ansEigenvalues[j];

  for(matrix::size_type j=0; j<nvars; j++)
    if(ansEigenvalues[j]<maxEigenvalue*1.e-16)
      ansEigenvalues[j]=maxEigenvalue*1.e-16; // all eigenvalues should be positive; any negative eigenvalues are assumed to result from roundoff error from zero

  determinant = 1.;
  for(matrix::size_type j=0; j<nvars; j++)
    determinant *= ansEigenvalues[j];
  matrix inv = matrix(nvars,nvars);
  for(matrix::size_type j=0; j<nvars; j++)
    for(matrix::size_type k=0; k<nvars; k++)
      {
	inv[j][k]=0.;
	for(matrix::size_type l=0; l<nvars; l++)
	  {
	    inv[j][k]+=
	      ansEigenvectors[j][l]*ansEigenvectors[k][l]*
	      pow(ansEigenvalues[l],-1.);
	  }
      }
  return(inv);
}


std::vector< std::vector<double> > matrix::toSTLVector()
{
  return(data);
}


#define NRANSI
#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

void jacobi(matrix & a, matrix::size_type n, vector<double> & d, matrix & v)
{
  matrix aa = a;
  int j,iq,i;
  double tresh,theta,tau,t,sm,s,h,g,c;

  vector<double> b(n);
  vector<double> z(n);
  for (matrix::size_type ip=0;ip<n;ip++) {
    for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (matrix::size_type ip=0;ip<n;ip++) {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  int nrot=0;
  for (i=0;i<50;i++) {
    sm=0.0;
    for (matrix::size_type ip=0;ip<n-1;ip++) {
      for (matrix::size_type iq=ip+1;iq<n;iq++)
	sm += fabs(a[ip][iq]);
    }
    if (sm == 0.0) {
      return;
    }
    if (i < 4)
      tresh=0.2*sm/(n*n);
    else
      tresh=0.0;
    for (matrix::size_type ip=0;ip<n-1;ip++) {
      for (matrix::size_type iq=ip+1;iq<n;iq++) {
	g=100.0*fabs(a[ip][iq]);
	if (i > 4 && (double)(fabs(d[ip])+g) == (double)fabs(d[ip])
	    && (double)(fabs(d[iq])+g) == (double)fabs(d[iq]))
	  a[ip][iq]=0.0;
	else if (fabs(a[ip][iq]) > tresh) {
	  h=d[iq]-d[ip];
	  if ((double)(fabs(h)+g) == (double)fabs(h))
	    t=(a[ip][iq])/h;
	  else {
	    theta=0.5*h/(a[ip][iq]);
	    t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
	    if (theta < 0.0) t = -t;
	  }
	  c=1.0/sqrt(1+t*t);
	  s=t*c;
	  tau=s/(1.0+c);
	  h=t*a[ip][iq];
	  z[ip] -= h;
	  z[iq] += h;
	  d[ip] -= h;
	  d[iq] += h;
	  a[ip][iq]=0.0;
	  for (j=0;j<=ip-1;j++) {
	    ROTATE(a,j,ip,j,iq)
	      }
	  for (j=ip+1;j<=iq-1;j++) {
	    ROTATE(a,ip,j,j,iq)
	      }
	  for (j=iq+1;j<n;j++) {
	    ROTATE(a,ip,j,iq,j)
	      }
	  for (j=0;j<n;j++) {
	    ROTATE(v,j,ip,j,iq)
	      }
	  ++(nrot);
	}
      }
    }
    for (matrix::size_type ip=0;ip<n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  cout << "Too many iterations in routine jacobi" << endl;
  cout << "This is the offending matrix:" << endl;
  aa.print();
  exit(1);
}

#undef ROTATE
#undef NRANSI
