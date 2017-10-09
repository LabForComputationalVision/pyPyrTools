#include "convolve.c"

void pprint_matrix(int x_dim, int y_dim, double* img) {
  int j, k;

  double mmin=1e10, mmax=-1e10;
  
  for(j=0; j < y_dim; j++) 
    for(k=0; k < x_dim; k++) {
      mmin = fmin(mmin, img[j * x_dim + k]);
      mmax = fmax(mmax, img[j * x_dim + k]);
    }

  for(j=0; j < y_dim; j++) {
    for(k=0; k < x_dim; k++) {
      printf("%d ", (int)(128 + 127 * (img[j * x_dim + k])/fmax(fabs(mmax),fabs(mmin))));
    }
    printf("\n");
  }
}

int main()
{

  double filt[] = {
    0.1,0.1,0.0,
    -0.2,0.3,-0.2,
    0.1,0.1,0.1
  };

  double image[] = {
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0.1,0,0,0,
    0,0,0,0.2,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
  };
  /* double image[] = { */
  /*   0.1,0.2,0.3,0.4,0.5,0.6, */
  /*   0.1,0.2,0.4,0.5,0.6,0.7, */
  /*   0.1,0.2,0.3,0.4,0.5,0.6, */
  /*   0.1,0.2,0.4,0.5,0.0,0.7, */
  /*   0.1,0.6,0.3,0.4,0.5,0.6, */
  /*   0.1,0.3,0.4,0.9,0.6,0.7 */
  /* }; */
  
  double temp[] = {
    0,0,0,  
    0,0,0,
    0,0,0
  };
  
  int x_fdim=3, x_dim=6;
  int y_fdim=3, y_dim=6;
  int x_step=1, y_step=1;
  int x_start=0, y_start=0;
  int x_stop=6, y_stop=6;
  
  char edges[] = "reflect1";

  double result[] = {
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0
  };
  
  int qq = internal_expand(image, filt, temp,
			   x_fdim, y_fdim,
			   x_start, x_step, x_stop, y_start, y_step, y_stop,
			   result, x_dim, y_dim, edges);
  int j, k;
  pprint_matrix(x_fdim, y_fdim, filt);
  printf("\n");
  pprint_matrix(x_dim, y_dim, image);
  printf("\n");
  pprint_matrix(x_dim, y_dim, result);
  printf("\n");

  return 0;
}


