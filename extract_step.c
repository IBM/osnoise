/* Copyright IBM Corporation, 2020
 * author : Bob Walkup
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char * argv[])
{
   int iter, cc, fd, nranks, myrank, maxiter;
   float * step;
   ssize_t rc;
   size_t nbytes;
   off_t offset, rs;
   char filename[80];
   FILE * ofp;

   if (argc != 2) {
     printf("syntax : ./extract_step osnoise.data\n");
     exit(0);
   }

   fd = open(argv[1], O_RDONLY);
   if (fd < 0) {
     printf("failed to open %s ... exiting\n", argv[1]);
     exit(0);
   }

   rc = read(fd, &nranks, 4);
   if (rc < 0) {
     printf("read failed for %s ... exiting\n", argv[1]);
     exit(0);
   }
  
   rc = read(fd, &maxiter, 4);
   if (rc < 0) {
     printf("read failed for %s ... exiting\n", argv[1]);
     exit(0);
   }

   printf("enter the rank that you want to extract : ");
   scanf("%d", &myrank);

   offset = 8L + 8L*((long) maxiter)*((long) myrank) + 4L*((long) maxiter);

   rs = lseek(fd, offset, SEEK_SET);
   if (rs < 0) {
     printf("lseek failed for %s ... exiting\n", argv[1]);
     exit(0);
   }
  
   step = (float *) malloc(maxiter*sizeof(float));
   if (step == NULL)  {
     printf("malloc failed for the step time array ... exiting\n");
     exit(0);
   }

   nbytes = maxiter*sizeof(float);

   rc = read(fd, step, nbytes);
   if (rc != nbytes) {
     printf("read failed for %s ... exiting\n", argv[1]);
     exit(0);
   } 

   cc = close(fd);
   if (cc < 0) {
     printf("close failed for %s ... exiting\n", argv[1]);
     exit(0);
   } 

   printf("extracting step times for rank %d ...\n", myrank);

   sprintf(filename, "%s.step.%d", argv[1], myrank);

   ofp = fopen(filename, "w");
   if (ofp == NULL) {
      printf("failed to open the output file %s ... exiting\n", filename);
      exit(0);
   }

   // write output in units of milliseconds
   for (iter = 0; iter < maxiter; iter++) fprintf(ofp, "%.6e\n", 1.0e3*step[iter]);

   fclose(ofp);

   printf("wrote file %s ... timing data in units of milliseconds\n", filename);

   return 0;
}
