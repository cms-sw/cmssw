
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/
#define TRACE //std::cout<<__FILE__<<" : "<<__LINE__<<std::endl;

#ifndef CIRCLE_FIT_H
#define CIRCLE_FIT_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

	struct Circle{
		double x;
		double y;
		double r;
		unsigned int n;
		double chi2;
	};


class CircleFit {

	public:
		CircleFit(){ mData.clear(); }
		CircleFit( const std::vector< GlobalPoint > &aData ) : mData(aData) {}
		~CircleFit(){}

		void clear()
		{
			mData.clear();
		}

		void push_back( const GlobalPoint& a )
		{
			mData.push_back( a );
		}


		Circle perpendicularBisectorFit(){
TRACE
			Circle circle;
			circle.x=0.0;
			circle.y=0.0;
			circle.r=0.0;
			circle.n = mData.size();
			circle.chi2=0.0;

			if (circle.n< 3) return circle;
TRACE

			//double x[20];
			//double y[20];	
			//double x2[20];
			//double y2[20];	
			//double X[20][20];
			//double Y[20][20];

			std::vector<double> x;
			std::vector<double> y;	
			std::vector<double> x2;
			std::vector<double> y2;
			std::vector< std::vector<double> > X;
			std::vector< std::vector<double> > Y;

			x.resize(circle.n);
			y.resize(circle.n);
			x2.resize(circle.n);
			y2.resize(circle.n);

			X.resize(circle.n);
			Y.resize(circle.n);
			for(unsigned int i = 0; i != circle.n; i++){
				X[i].resize(circle.n);
				Y[i].resize(circle.n);
			}

TRACE

			for(unsigned int i = 0; i != circle.n; i++){
				x[i] = mData[i].x();	
				y[i] = mData[i].y();	
				x2[i] = x[i] * x[i];	
				y2[i] = y[i] * y[i];	
				for(unsigned int j = 0; j != circle.n; j++){
					X[i][j] = mData[i].x() - mData[j].x();
					Y[i][j] = mData[i].y() - mData[j].y();					
				}
			}

TRACE

			for(unsigned int i = 0; i != circle.n-2; ++i){
				for(unsigned int j = i+1; j != circle.n-1; ++j){
					for(unsigned int k = j+1; k != circle.n; ++k){
						//std::cout << __FILE__ << " : " << __LINE__ << " " << i << " " << j << " " << k << " " << std::endl;
						double w	= (x[i]*Y[j][k]) + (x[j]*Y[k][i]) + (x[k]*Y[i][j]);
						double wb	= (x2[i]*Y[j][k]) + (x2[j]*Y[k][i]) + (x2[k]*Y[i][j]);
						double z	= (y[i]*X[j][k]) + (y[j]*X[k][i]) + (y[k]*X[i][j]);
						double zb	= (y2[i]*X[j][k]) + (y2[j]*X[k][i]) + (y2[k]*X[i][j]);
						double Xb	= X[i][j]*X[j][k]*X[k][i];
						double Yb	= Y[i][j]*Y[j][k]*Y[k][i];

						circle.x+=( (wb-Yb)/w );
						circle.y+=( (zb-Xb)/z );
					}
				}
			}


TRACE

			double scale = circle.n * (circle.n-1) * (circle.n-2);
			scale = 3.0 / scale;

			circle.x*=scale;
			circle.y*=scale;

TRACE

			for(unsigned int i = 0; i != circle.n; i++){
				double dx = (x[i]-circle.x);
				double dy = (y[i]-circle.y);
				circle.r += sqrt( (dx*dx) + (dy*dy) );
			}
			circle.r /= (double) circle.n;

TRACE

			for(unsigned int i = 0; i != circle.n; i++){
				double dx = (x[i]-circle.x);
				double dy = (y[i]-circle.y);
				double dr = sqrt( (dx*dx) + (dy*dy) ) - circle.r;
				circle.chi2 += dr * dr / circle.r;
			}
TRACE


			return circle;
		}



		Circle modifiedLeastSquaresFit(){
TRACE
			//std::cout << __FILE__ << " : " << __LINE__ << std::endl;
			Circle circle;
			circle.x=0.0;
			circle.y=0.0;
			circle.r=0.0;
			circle.n = mData.size();
			circle.chi2=0.0;

			if (circle.n< 3) return circle;

			double sumx = 0.0;
			double sumx2 = 0.0;
			double sumx3 = 0.0;
			double sumy = 0.0;
			double sumy2 = 0.0;
			double sumy3 = 0.0;
			double sumxy = 0.0;
			double sumx2y = 0.0;
			double sumxy2 = 0.0;
TRACE
			for(unsigned int i = 0; i != circle.n; i++){
				double x = mData[i].x();
				double y = mData[i].y();

				sumx += x;
				sumx2 += (x*x);
				sumx3 += (x*x*x);
				sumy += y;
				sumy2 += (y*y);
				sumy3 += (y*y*y);
				sumxy += (x*y);
				sumx2y += (x*x*y);
				sumxy2 += (x*y*y);
			}
			double n = (double) circle.n;
TRACE

			double A = (n*sumx2)-(sumx*sumx);
			double B = (n*sumxy)-(sumx*sumy);
			double C = (n*sumy2)-(sumy*sumy);

			double D = (n*(sumxy2+sumx3)) - (sumx*(sumy2+sumx2)) ;
			D/=2;
			double E = (n*(sumx2y+sumy3)) - (sumy*(sumx2+sumy2)) ;
			E/=2;

TRACE
			circle.x = (D*C)-(B*E) ;
			circle.y = (A*E)-(B*D) ;

TRACE
			double scale = (A*C)-(B*B);
			scale = 1.0/scale;

			circle.x*=scale;
			circle.y*=scale;

TRACE

			for(unsigned int i = 0; i != circle.n; i++){
				double dx = (mData[i].x()-circle.x);
				double dy = (mData[i].y()-circle.y);
				circle.r += sqrt( (dx*dx) + (dy*dy) );
			}
			circle.r /= n;

			//std::cout << __FILE__ << " : " << __LINE__ << std::endl;
TRACE

			for(unsigned int i = 0; i != circle.n; i++){
				double dx = (mData[i].x()-circle.x);
				double dy = (mData[i].y()-circle.y);
				double dr = sqrt( (dx*dx) + (dy*dy) ) - circle.r;
				circle.chi2 += dr * dr / circle.r;
			}
TRACE

			return circle;

		}

	private:
		std::vector< GlobalPoint > mData;
};
#endif

