/*				SplitBregman3DROF.c (MEX version) by Mingxin Jin
 *   This code is written on the basis of the SplitBregmanROF.c by Tom Goldstein.
 *   This code performs anisotropic 3D ROF denoising using the "Split Bregman" algorithm.
 * This version of the code has a "mex" interface, and should be compiled and called
 * through MATLAB.
 *
 *DISCLAIMER:  This code is for academic (non-commercial) use only.  Also, this code
 *comes with absolutely NO warranty of any kind: I do my best to write reliable codes,
 *but I take no responsibility for the reliability of the results.
 *
 *                      HOW TO COMPILE THIS CODE
 *   To compile this code, open a MATLAB terminal, and change the current directory to
 *the folder where this "c" file is contained.  Then, enter this command:
 *    >>  mex splitBregman3DROF.c
 *This file has been tested under windows using visual studio, and under linux mint using gcc.
 *Once the file is compiled, the command "splitBregman3DROF" can be used just like any
 *other MATLAB m-file.
 *
 *                      HOW TO USE THIS CODE
 * An image is denoised using the following command
 *
 *   SplitBregmanROF(video,mu,tol,cols,rows);
 *
 * where:
 *   - "video" is a 2d array containing vectorized images.
 *   - "mu" is the weighting parameter for the fidelity term
 *   - "tol" is the stopping tolerance for the iteration.  "tol"=0.001 is reasonable for
 *            most applications.
 *   - [rows,cols] equals the image size in matlab.
 *
 */

#include <math.h>
#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
typedef double num;


/*A method for isotropic TV*/
void rof_iso(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz,
             double mu, double lambda, int nGS, int nBreg, int width, int height, int height1, int height2);

/*A method for Anisotropic TV*/
void rof_an(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz,
            double mu, double lambda, int nGS, int nBreg, int width, int height, int height1, int height2);

/*****************Minimization Methods*****************/
void gs_an(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height, int height1, int height2, int iter);
void gs_iso(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height, int height1, int height2, int iter);

/******************Relaxation Methods*****************/
void gsU(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height, int height1, int height2);
void gsX(num** u, num** x, num** bx , double lambda, int width, int height, int height1, int height2);
void gsY(num** u, num** y, num** by , double lambda, int width, int height, int height1, int height2);
void gsZ(num** u, num** z, num** bz , double lambda, int width, int height, int height1, int height2);
void gsSpace(num** u, num** x, num** y, num** z, num** bx, num** by, num** bz, double lambda, int width, int height, int height1, int height2);

/************************Bregman***********************/
void bregmanX(num** x,num** u, num** bx, int width, int height, int height1, int height2);
void bregmanY(num** y,num** u, num** by, int width, int height, int height1, int height2);
void bregmanZ(num** z,num** u, num** bz, int width, int height, int height1, int height2);

/**********************Memory************/

num** newMatrix(int rows, int cols);
void deleteMatrix(num ** a);
double** get2dArray(const mxArray *mx, int isCopy);
double copy(double** source, double** dest, int rows, int cols);

/***********************The MEX Interface to rof_iso()***************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* get the size of the image*/
	int rows = mxGetN(prhs[0]);
	int cols = mxGetM(prhs[0]);
	int col1 = (double)(mxGetScalar(prhs[3]));
	int col2 = (double)(mxGetScalar(prhs[4]));

	/* get the fidelity and convergence parameters*/
	double mu =  (double)(mxGetScalar(prhs[1]));
	double lambda = 2*mu;
	double tol = (double)(mxGetScalar(prhs[2]));

	/* get the image, and declare memory to hold the auxillary variables*/
	double **f = get2dArray(prhs[0],0);
	double **u = newMatrix(rows,cols);
	double **x = newMatrix(rows,cols);
	double **y = newMatrix(rows,cols);
	double **z = newMatrix(rows,cols);
	double **bx = newMatrix(rows,cols);
	double **by = newMatrix(rows,cols);
	double **bz = newMatrix(rows,cols);

	double** uOld;
	double *outArray;
	double diff;
	int count;
	int i,j;

	/***********Check Conditions******/
	if (nrhs != 5)
	{
		mexErrMsgTxt("Five input arguments required.");
	}
	if (nlhs > 1)
	{
		mexErrMsgTxt("Too many output arguments.");
	}
	if (!(mxIsDouble(prhs[0])))
	{
		mexErrMsgTxt("Input array must be of type double.");
	}

	/* Use a copy of the image as an initial guess*/
	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
		{
			u[i][j] = f[i][j];
		}
	}

	/* perform iterations*/

	uOld = newMatrix(rows,cols);
	count=0;
	do
	{
		rof_an(u,f,x,y,z,bx,by,bz,mu,lambda,5,1,rows,cols,col1,col2);
		diff = copy(u,uOld,rows,cols);
		count++;
	}
	while( (diff>tol && count<500) || count<5 );

	/* copy to output vector*/
	plhs[0] = mxCreateDoubleMatrix(cols, rows, mxREAL); /*mxReal is our data-type*/
	outArray = mxGetPr(plhs[0]);
	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
		{
			outArray[(i*cols)+j] = u[i][j];
		}
	}

	/* Free memory */
	deleteMatrix(u);
	deleteMatrix(x);
	deleteMatrix(y);
	deleteMatrix(z);
	deleteMatrix(bx);
	deleteMatrix(by);
	deleteMatrix(bz);
	deleteMatrix(uOld);

	return;
}

double** get2dArray(const mxArray *mx, int isCopy)
{
	double* oned = mxGetPr(mx);
	int rowLen = mxGetN(mx);
	int colLen = mxGetM(mx);
	double** rval = (double**) malloc(rowLen*sizeof(double*));
	int r;
	if(isCopy)
	{
		double *copy = (double*)malloc(rowLen*colLen*sizeof(double));
		int i, sent = rowLen*colLen;
		for(i=0; i<sent; i++)
			copy[i]=oned[i];
		oned=copy;
	}

	for(r=0; r<rowLen; r++)
		rval[r] = &oned[colLen*r];
	return rval;
}







/*                IMPLEMENTATION BELOW THIS LINE                         */

/******************Isotropic TV**************/

void rof_iso(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz,
             double mu, double lambda, int nGS, int nBreg, int width, int height,
             int height1, int height2)
{
	/*int breg;
	//for(breg=0;breg<nBreg;breg++){*/
	gs_iso(u,f,x,y,z,bx,by,bz,mu,lambda,width, height,height1,height2,nGS);
	bregmanX(x,u,bx,width,height,height1,height2);
	bregmanY(y,u,by,width,height,height1,height2);
	bregmanZ(z,u,bz,width,height,height1,height2);
	/*}*/
}


void gs_iso(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height,
            int height1, int height2, int iter)
{
	int j;
	for(j=0; j<iter; j++)
	{
		gsU(u,f,x,y,z,bx,by,bz,mu,lambda,width,height,height1,height2);
	}
	gsSpace(u,x,y,z,bx,by,bz,lambda,width,height,height1,height2);


}


/******************Anisotropic TV**************/
void rof_an(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz,
            double mu, double lambda, int nGS, int nBreg, int width, int height,
            int height1, int height2)
{
	/*int breg;
	//for(breg=0;breg<nBreg;breg++){*/
	gs_an(u,f,x,y,z,bx,by,bz,mu,lambda,width, height,height1,height2,nGS);
	bregmanX(x,u,bx,width,height,height1,height2);
	bregmanY(y,u,by,width,height,height1,height2);
	bregmanZ(z,u,bz,width,height,height1,height2);

}


void gs_an(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height,
           int height1, int height2, int iter)
{
	int j;
	for(j=0; j<iter; j++)
	{
		gsU(u,f,x,y,z,bx,by,bz,mu,lambda,width,height,height1,height2);
	}
	gsX(u,x,bx,lambda,width,height,height1,height2);
	gsY(u,y,by,lambda,width,height,height1,height2);
	gsZ(u,z,bz,lambda,width,height,height1,height2);

}





/****Relaxation operators****/

void gsU(num** u, num** f, num** x, num** y, num** z, num** bx, num** by, num** bz, double mu, double lambda, int width, int height,
         int height1, int height2)
{
	int w,h1,h2;
	double sum;
	int wm1,hlen,h1h2,h1h2m1;
	double normConst = 1.0/(mu+6*lambda);
	int wSent = width-1, h1Sent = height1-1, h2Sent = height2-1;
	for(w=1; w<wSent; w++) 		/* do the central pixels*/
	{
		wm1 = w-1;
		for(h1=1; h1<h1Sent; h1++)
		{
			hlen = h1*height2;
			for(h2=1; h2<h2Sent; h2++)
			{
				h1h2=hlen+h2;
				h1h2m1=h1h2-1;
				sum = x[wm1][h1h2] - x[w][h1h2]+y[w][h1h2-height2] - y[w][h1h2]+z[w][h1h2m1]-z[w][h1h2]
				      -bx[wm1][h1h2] + bx[w][h1h2]-by[w][h1h2-height2] + by[w][h1h2] - bz[w][h1h2m1]+bz[w][h1h2];
				sum+=(u[w+1][h1h2]+u[wm1][h1h2]+u[w][h1h2-height2]+u[w][h1h2+height2]+u[w][h1h2m1]+u[w][h1h2+1]);
				sum*=lambda;
				sum+=mu*f[w][h1h2];
				sum*=normConst;
				u[w][h1h2] = sum;
			}
			h2=0;	/*do the h2 bottom*/
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = x[wm1][h1h2] - x[w][h1h2]+y[w][h1h2-height2] - y[w][h1h2]-z[w][h1h2]
			      -bx[wm1][h1h2] + bx[w][h1h2]-by[w][h1h2-height2] + by[w][h1h2]+bz[w][h1h2];
			sum+=(u[w+1][h1h2]+u[wm1][h1h2]+u[w][h1h2-height2]+u[w][h1h2+height2]+u[w][h1h2+1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
			h2=h2Sent;	/*do the h2 top*/
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = x[wm1][h1h2] - x[w][h1h2]+y[w][h1h2-height2] - y[w][h1h2]+z[w][h1h2m1]
			      -bx[wm1][h1h2] + bx[w][h1h2]-by[w][h1h2-height2] + by[w][h1h2] - bz[w][h1h2m1];
			sum+=(u[w+1][h1h2]+u[wm1][h1h2]+u[w][h1h2-height2]+u[w][h1h2+height2]+u[w][h1h2m1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
		}
		h1=0;	/*do the left pixels*/
		hlen = h1*height2;
		for(h2=1; h2<h2Sent; h2++)
		{
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = x[wm1][h1h2] - x[w][h1h2] - y[w][h1h2]+z[w][h1h2m1]-z[w][h1h2]
			      -bx[wm1][h1h2] + bx[w][h1h2] + by[w][h1h2] - bz[w][h1h2m1]+bz[w][h1h2];
			sum+=(u[w+1][h1h2]+u[wm1][h1h2]+u[w][h1h2+height2]+u[w][h1h2m1]+u[w][h1h2+1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
		}
		h1=h1Sent;  /*do the right pixels*/
		hlen = h1*height2;
		for(h2=1; h2<h2Sent; h2++)
		{
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = x[wm1][h1h2] - x[w][h1h2]+y[w][h1h2-height2]+z[w][h1h2m1]-z[w][h1h2]
			      -bx[wm1][h1h2] + bx[w][h1h2]-by[w][h1h2-height2] - bz[w][h1h2m1]+bz[w][h1h2];
			sum+=(u[w+1][h1h2]+u[wm1][h1h2]+u[w][h1h2-height2]+u[w][h1h2m1]+u[w][h1h2+1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
		}
	}
	w=0;    /*do the first frame pixels*/
	wm1 = w-1;
	for(h1=1; h1<h1Sent; h1++)
	{
		hlen = h1*height2;
		for(h2=1; h2<h2Sent; h2++)
		{
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = - x[w][h1h2]+y[w][h1h2-height2] - y[w][h1h2]+z[w][h1h2m1]-z[w][h1h2]
			      + bx[w][h1h2]-by[w][h1h2-height2] + by[w][h1h2] - bz[w][h1h2m1]+bz[w][h1h2];
			sum+=(u[w+1][h1h2]+u[w][h1h2-height2]+u[w][h1h2+height2]+u[w][h1h2m1]+u[w][h1h2+1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
		}
	}
	w=wSent;    /*do the last frame pixels*/
	wm1 = w-1;
	for(h1=1; h1<h1Sent; h1++)
	{
		hlen = h1*height2;
		for(h2=1; h2<h2Sent; h2++)
		{
			h1h2=hlen+h2;
			h1h2m1=h1h2-1;
			sum = x[wm1][h1h2]+y[w][h1h2-height2] - y[w][h1h2]+z[w][h1h2m1]-z[w][h1h2]
			      -bx[wm1][h1h2]-by[w][h1h2-height2] + by[w][h1h2] - bz[w][h1h2m1]+bz[w][h1h2];
			sum+=(u[wm1][h1h2]+u[w][h1h2-height2]+u[w][h1h2+height2]+u[w][h1h2m1]+u[w][h1h2+1]);
			sum*=lambda;
			sum+=mu*f[w][h1h2];
			sum/=mu+5*lambda;
			u[w][h1h2] = sum;
		}
	}
}




	void gsSpace(num** u, num** x, num** y, num** z, num** bx, num** by, num** bz, double lambda, int width, int height,
	             int height1, int height2)
	{
		int w,h1,h2;
		num a,b,c,s;
		num flux = 1.0/lambda;
		num mflux = -1.0/lambda;
		num flux2 = flux*flux;
		num *uw,*uwp1,*bxw,*byw,*bzw,*xw,*yw,*zw;
		/*num base;*/
		int hlen,h1h2;
		for(w=0; w<width-1; w++)
		{
			uw = u[w];
			uwp1=u[w+1];
			bxw=bx[w];
			byw=by[w];
			bzw=bz[w];
			xw=x[w];
			yw=y[w];
			zw=z[w];
			for(h1=0; h1<height1-1; h1++)
			{
				hlen = h1*height2;
				for(h2=0; h2<height2-1; h2++)
				{
					h1h2=hlen+h2;
					a = uwp1[h1h2]-uw[h1h2]+bxw[h1h2];
					b = uw[h1h2+height2]-uw[h1h2]+byw[h1h2];
					c = uw[h1h2+1]-uw[h1h2]+bzw[h1h2];
					s = a*a+b*b+c*c;
					if(s<flux2)
					{
						xw[h1h2]=0;
						yw[h1h2]=0;
						zw[h1h2]=0;
						continue;
					}
					s = sqrt(s);
					s=(s-flux)/s;
					xw[h1h2] = s*a;
					yw[h1h2] = s*b;
					zw[h1h2] = s*c;
				}
			}
			h1=height1-1;
			hlen = h1*height2;
			for(h2=0; h2<height2-1; h2++)
			{
				h1h2=hlen+h2;
				a = uwp1[h1h2]-uw[h1h2]+bxw[h1h2];
				c = uw[h1h2+1]-uw[h1h2]+bzw[h1h2];
				s = a*a+c*c;
				if(s<flux2)
				{
					xw[h1h2]=0;
					zw[h1h2]=0;
					continue;
				}
				s = sqrt(s);
				s=(s-flux)/s;
				xw[h1h2] = s*a;
				zw[h1h2] = s*c;
			}
			h2 = height2-1;
			for(h1=0; h1<height1-1; h1++)
			{
				h1h2=h1*height2+h2;
				a = uwp1[h1h2]-uw[h1h2]+bxw[h1h2];
				b = uw[h1h2+height2]-uw[h1h2]+byw[h1h2];
				s = a*a+b*b;
				if(s<flux2)
				{
					xw[h1h2]=0;
					yw[h1h2]=0;
					continue;
				}
				s = sqrt(s);
				s=(s-flux)/s;
				xw[h1h2] = s*a;
				yw[h1h2] = s*b;
			}
		}
		w=width-1;
		for(h1=0; h1<height1-1; h1++)
		{
			hlen = h1*height2;
			for(h2=0; h2<height2-1; h2++)
			{
				h1h2=hlen+h2;
				b = uw[h1h2+height2]-uw[h1h2]+byw[h1h2];
				c = uw[h1h2+1]-uw[h1h2]+bzw[h1h2];
				s = b*b+c*c;
				if(s<flux2)
				{
					yw[h1h2]=0;
					zw[h1h2]=0;
					continue;
				}
				s = sqrt(s);
				s=(s-flux)/s;
				yw[h1h2] = s*b;
				zw[h1h2] = s*c;
			}
		}
	}


	void gsX(num** u, num** x, num** bx , double lambda, int width, int height, int height1, int height2)
	{
		int w,h;
		double base;
		const double flux = 1.0/lambda;
		const double mflux = -1.0/lambda;
		num* uwp1;
		num* uw;
		num* bxw;
		num* xw;
		width = width-1;
		for(w=0; w<width; w++)
		{
			uwp1 = u[w+1];
			uw = u[w];
			bxw = bx[w];
			xw = x[w];
			for(h=0; h<height; h++)
			{
				base = uwp1[h]-uw[h]+bxw[h];
				if(base>flux)
				{
					xw[h] = base-flux;
					continue;
				}
				if(base<mflux)
				{
					xw[h] = base+flux;
					continue;
				}
				xw[h] = 0;
			}
		}
	}

	void gsY(num** u, num** y, num** by , double lambda, int width, int height, int height1, int height2)
	{
		int w,h1,h2;
		double base;
		const double flux = 1.0/lambda;
		const double mflux = -1.0/lambda;
		num* uw;
		num* yw;
		num* bw;
		int hlen,h1h2;
		height1 = height1-1;
		for(w=0; w<width; w++)
		{
			uw = u[w];
			yw = y[w];
			bw = by[w];
			for(h1=0; h1<height1; h1++)
			{
				hlen = h1*height2;
				for(h2=0; h2<height2; h2++)
				{
					h1h2=hlen+h2;
					base = uw[h1h2+height2]-uw[h1h2]+bw[h1h2];
					if(base>flux)
					{
						yw[h1h2] = base-flux;
						continue;
					}
					if(base<mflux)
					{
						yw[h1h2] = base+flux;
						continue;
					}
					yw[h1h2] = 0;
				}
			}
		}
	}

	void gsZ(num** u, num** z, num** bz , double lambda, int width, int height, int height1, int height2)
	{
		int w,h1,h2;
		double base;
		const double flux = 1.0/lambda;
		const double mflux = -1.0/lambda;
		num* uw;
		num* zw;
		num* bw;
		int hlen,h1h2;
		height2 = height2-1;
		for(w=0; w<width; w++)
		{
			uw = u[w];
			zw = z[w];
			bw = bz[w];
			for(h1=0; h1<height1; h1++)
			{
				hlen = h1*height2;
				for(h2=0; h2<height2; h2++)
				{
					h1h2=hlen+h2;
					base = uw[h1h2+1]-uw[h1h2]+bw[h1h2];
					if(base>flux)
					{
						zw[h1h2] = base-flux;
						continue;
					}
					if(base<mflux)
					{
						zw[h1h2] = base+flux;
						continue;
					}
					zw[h1h2] = 0;
				}
			}
		}
	}

	void bregmanX(num** x,num** u, num** bx, int width, int height, int height1, int height2)
	{
		int w,h;
		double d;
		num* uwp1,*uw,*bxw,*xw;
		for(w=0; w<width-1; w++)
		{
			uwp1=u[w+1];
			uw=u[w];
			bxw=bx[w];
			xw=x[w];
			for(h=0; h<height; h++)
			{
				d = uwp1[h]-uw[h];
				bxw[h]+= d-xw[h];
			}
		}
	}


	void bregmanY(num** y,num** u, num** by, int width, int height, int height1, int height2)
	{
		int w,h1,h2;
		double d;
		int h1Sent = height1-1;
		num* uw,*byw,*yw;
		int hlen,h1h2;
		for(w=0; w<width; w++)
		{
			uw=u[w];
			byw=by[w];
			yw=y[w];
			for(h1=0; h1<h1Sent; h1++)
			{
				hlen = h1*height2;
				for(h2=0; h2<height2; h2++)
				{
					h1h2 = hlen+h2;
					d = uw[h1h2+height2]-uw[h1h2];
					byw[h1h2]+=d-yw[h1h2];
				}
			}
		}
	}

	void bregmanZ(num** z,num** u, num** bz, int width, int height, int height1, int height2)
	{
		int w,h1,h2;
		double d;
		int h2Sent = height2-1;
		num* uw,*bzw,*zw;
		int hlen,h1h2;
		for(w=0; w<width; w++)
		{
			uw=u[w];
			bzw=bz[w];
			zw=z[w];
			for(h1=0; h1<height1; h1++)
			{
				hlen = h1*height2;
				for(h2=0; h2<h2Sent; h2++)
				{
					h1h2 = hlen+h2;
					d = uw[h1h2+1]-uw[h1h2];
					bzw[h1h2]+=d-zw[h1h2];
				}
			}
		}
	}


	/************************memory****************/

	double copy(double** source, double** dest, int rows, int cols)
	{
		int r,c;
		double temp,sumDiff, sum;
		for(r=0; r<rows; r++)
			for(c=0; c<cols; c++)
			{
				temp = dest[r][c];
				sum+=temp*temp;
				temp -= source[r][c];
				sumDiff +=temp*temp;

				dest[r][c]=source[r][c];

			}
		return sqrt(sumDiff/sum);
	}



	num** newMatrix(int rows, int cols)
	{
		num* a = (num*) malloc(rows*cols*sizeof(num));
		num** rval = (num**) malloc(rows*sizeof(num*));
		int j,g;
		rval[0] = a;
		for(j=1; j<rows; j++)
			rval[j] = &a[j*cols];

		for(j=0; j<rows; j++)
			for(g=0; g<cols; g++)
				rval[j][g] = 0;
		return rval;
	}
	void deleteMatrix(num ** a)
	{
		free(a[0]);
		free(a);
	}
