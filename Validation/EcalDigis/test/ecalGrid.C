void ruGrid(int mode = 1111){
  //EE-
  if(mode >= 1000) eeGrid(-39.5, .5, .2);
  //EB-
  if((mode % 1000) >= 100) ebGrid(-17.5, .5, .2, 10);
  //EB+
  if((mode % 100) >= 10) ebGrid(.5, .5, .2, 10);
  //EE+
  if((mode % 10) >= 1) eeGrid(20.5, .5, .2);
}

void xtalGrid(int mode = 1111){
  //EE-
  if(mode >= 1000) eeGrid(-199.5, .5, 1.);
  //EB-
  if((mode % 1000) >= 100) ebGrid(-85.5, .5, 1.);
  //EB+
  if((mode % 100) >= 10) ebGrid(.5, .5, 1.);
  //EE+
  if((mode % 10) >= 1) eeGrid(100.5, .5, 1.);
}

void tccGrid(int mode = 1111){
  //EE-
  if(mode >= 1000) drawGrid(-28.5, 0.5, -21.5, 72.5, 1, 18, 2);
  if(mode >= 1000) drawGrid(-21.5, 0.5, -17.5, 72.5, 1, 18, 2);
  //EB-
  if((mode % 1000) >= 100) drawGrid(-17.5, 0.5, -.5, 72.5, 1, 18, 2);
  //EB+
  if((mode % 100) >= 10) drawGrid(.5, 0.5, 17.5, 72.5, 1, 18, 2);
  //EE+
  if((mode % 10) >= 1) drawGrid(17.5, 0.5, 21.5, 72.5, 1, 18, 2);
  if((mode % 10) >= 1) drawGrid(21.5, 0.5, 28.5, 72.5, 1, 18, 2);
}

void ebGrid(float xOffset = .5, float yOffset = .5, float scale  = 1.,
            int phiOffset =0) {
  float x1 = 0 + xOffset;
  float y1 = 0 + yOffset;

  float x2 = 85*scale + xOffset;
  float y2 = 360*scale + yOffset;
  
  drawGrid(x1, y1, x2, y2, 1, 18, phiOffset*scale);
}

void eeGrid(float xOffset = .5, float yOffset = .5, float scale  = 1.){
  const unsigned n = 40;
  int x[n] = {  0,   3,  5,  8, 13, 15, 20, 25, 35, 40,
                60, 65, 75, 80, 85, 87, 92, 95, 97,100,
                97, 95, 92, 87, 85, 80, 75, 65, 60,
                40, 35, 25, 20, 15, 13,  8,  5,  3, 0, 0
  };
  int y[n] = { 40,  35, 25, 20, 15, 13,  8,  5,  3, 0
               0,    3,  5,  8, 13, 15, 20, 25, 35, 40,
               60,  65, 75, 80, 85, 87, 92, 95, 97,
               100, 97, 95, 92, 87, 85, 80, 75, 65, 60, 40
  };

  stairDraw(n, x, y, xOffset, yOffset, scale);

  const unsigned in = 24;
  int ix[in] = {39, 40, 41, 42, 43, 45, 55, 57, 58, 59, 60, 61,
                60, 59, 58, 57, 55, 45, 43, 42, 41, 40, 39, 39};
  int iy[in] = {45, 43, 42, 41, 40, 39, 40, 41, 42, 43, 45, 55,
                57, 58, 59, 60, 61, 60, 59, 58, 57, 55, 45, 45};

  stairDraw(in, ix, iy, xOffset, yOffset, scale);

  int x1[] = {61, 65, 90, 98};
  int y1[] = {50, 55, 60, 60};
  stairDraw(sizeof(x1)/sizeof(x1[0]), x1, y1, xOffset, yOffset, scale);
  
  int x2[] = {57, 60, 65, 70, 75, 80, 80};
  int y2[] = {60, 65, 70, 75, 85, 88, 88};
  stairDraw(sizeof(x2)/sizeof(x2[0]), x2, y2, xOffset, yOffset, scale);

  int x3[] = { 50 , 50, 50};
  int y3[] = { 61, 100, 100};
  stairDraw(sizeof(x3)/sizeof(x3[0]), x3, y3, xOffset, yOffset, scale);

  int x4[] = {39, 35, 10, 2};
  int y4[] = {50, 55, 60, 60};
  stairDraw(sizeof(x4)/sizeof(x4[0]), x4, y4, xOffset, yOffset, scale);
  
  int x5[] = {43, 40, 35, 30, 25, 20, 20};
  int y5[] = {60, 65, 70, 75, 85, 88, 88};
  stairDraw(sizeof(x5)/sizeof(x5[0]), x5, y5, xOffset, yOffset, scale);
  
  int x6[] = {61, 65, 70, 80, 90, 92};
  int y6[] = {45, 40, 35, 30, 25, 25};
  stairDraw(sizeof(x6)/sizeof(x6[0]), x6, y6, xOffset, yOffset, scale);
  
  int x7[] = {55, 55, 60, 65, 65};
  int y7[] = {39, 30, 15, 3, 3};
  stairDraw(sizeof(x7)/sizeof(x7[0]), x7, y7, xOffset, yOffset, scale);

  int x8[] = { 45, 45, 40, 35, 35};
  int y8[] = { 39, 30, 15, 3, 3};
  stairDraw(sizeof(x8)/sizeof(x8[0]), x8, y8, xOffset, yOffset, scale);
  
  int x9[] = {39, 35, 30, 20, 10,  8};
  int y9[] = {45, 40, 35, 30, 25, 25};
  stairDraw(sizeof(x9)/sizeof(x9[0]), x9, y9, xOffset, yOffset, scale); 
}

void stairDraw(unsigned n, int x[], int y[],
               float xOffset = .5, float yOffset = .5, float scale  = 1.){
  Float_t* xx = new Float_t[2*n-1];
  Float_t* yy = new Float_t[2*n-1];
  
  for(unsigned i = 0; i < n-1; ++i){
    xx[2*i] = x[i]*scale + xOffset;
    yy[2*i] = y[i]*scale + yOffset;
    xx[2*i+1] = x[i+1]*scale + xOffset;
    yy[2*i+1] = y[i]*scale + yOffset; 
  }
  //  xx[2*n-2] = xx[0];
  // yy[2*n-2] = yy[0];
  
  TPolyLine* p = new TPolyLine(2*n-2, xx, yy);
  p->Draw();
}


void drawGrid(float x1, float y1, float x2, float y2,
              int nxbins, int nybins, float yOffset = 0.){
  float step = (x2-x1) / nxbins;
  TLine* l = new TLine(0,0,0,0);
  for(float x = x1; x < x2; x += step){
    l->DrawLine(x, y1, x, y2);
  }
  l->DrawLine(x2, y1, x2, y2);

  step = (y2-y1) / nybins;

  for(float y = y1; y < y2; y += step){
    l->DrawLine(x1, y+yOffset, x2, y+yOffset);
  }
  if(yOffset<1.e-9) l->DrawLine(x1, y2, x2, y2);
}
