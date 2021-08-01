/* Copyright IBM Corporation, 2020
 * author : Bob Walkup
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>

#define BARRIER   10
#define EXCHANGE  11
#define ALLREDUCE 12
#define ALLTOALL  13

#define SORT_ASCENDING_ORDER   1
#define SORT_DESCENDING_ORDER -1

void compute(int, long, int, double *, double *);

double lutexp(double);

void exch(int);

void print_help(void);

void sortx(double *, int, int *, int);

static int myrank, myrow, mycol, npex, npey, dblcount;

static double * sendn,  * sende,  * sends,  * sendw,  * recvn,  * recve,  * recvs,  * recvw;



int main(int argc, char * argv[])
{
  int i, k, msgsize, nranks, ch, barrier_flag, help_flag = 0;
  int iter, maxiter, calib_iters, bin, numbpd, totbins, method, kernel;
  long npts, context_switches, context_switches_initial;
  long * all_context_switches;
  double  ssum, xtraffic, ytraffic, data_volume, bw;
  double  t1, t2, t3, tmin, tmax, delay;
  double * sbuf, * rbuf, * xrand;
  int nrand = 1000;
  int local_rank, ranks_per_node;
  MPI_Comm local_comm;

  double * tcomm, * tcomp, * tstep;
  double compute_interval_msec;
  double elapsed1, elapsed2, target_measurement_time;
  char format[160], hfmt[8], heading[160];

  double histo_bin_width, xhisto, hmin, hmax;
  double dnumbpd, log10_tmin, log10_tmax, topp, topn, botp, botn, exp1, exp2;
  double tavg, ssq, samples, relative_variation;
  double dev, sigma, scb, sp4, skewness, kurtosis;
  long * histo;
  struct rusage RU;
  struct minStruct {
                      double value;
                      int    index;
                   };
  struct minStruct myMPI, minMPI;
  double avgMPI;
  int mycpu, * all_cpus;
  int hostlen, maxlen;
  char host[80], * ptr, * snames, * rnames;
  int * sort_key;
  int num_nodes, color, key;
  double * compmax, compmin, compavg, compssq;
  double sigma_comp, samples_total, ssq_comp;
  double * aggregate_sigma_comp, * aggregate_compavg;
  double mean_comp, * alldev_comp, * allavg_comp;
  double sum_comm, * allsum_comm;
  double usr_cpu_initial, usr_cpu_final, usr_time, * allusr_time;
  double minavg, maxavg;
  int minrank, maxrank;
  int jobid, rank, green_light, dump_data, tag = 99;
  float * compbuf, * stepbuf, * floatcomp, * floatstep;
  char outfile[80];
  time_t current_time;
  char * time_string;
  FILE * ofp;
  MPI_Status status;
  MPI_Comm node_comm, column_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  MPI_Barrier(MPI_COMM_WORLD);
  current_time = time(NULL);
  time_string = ctime(&current_time);
  if (myrank == 0) fprintf(stderr, "starting time : %s\n", time_string);

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myrank, MPI_INFO_NULL, &local_comm);
  MPI_Comm_size(local_comm, &ranks_per_node);
  MPI_Comm_rank(local_comm, &local_rank);
  
  num_nodes = nranks / ranks_per_node;

  // create a communicator for collecting data across nodes
  color = local_rank;
  key = myrank;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &node_comm);

  if (myrank == 0) printf("checking program args ...\n");

  // set sensible default values then check program args
  numbpd = 10;  // number of histogram bins per decade
  compute_interval_msec = 3.0;
  target_measurement_time = 300.0;  // units of seconds
  msgsize = 100;
  method = EXCHANGE;
  kernel = 0;
  dump_data = 0;
  barrier_flag = 0;

  while (1) {
     ch = getopt(argc, argv, "hm:c:t:n:x:k:db");
     if (ch == -1) break;
     switch (ch) {
        case 'h':
           help_flag = 1;
           break;
        case 'b':  // optionally add infrequent barriers
           barrier_flag = 1;
           break;
        case 'c':  // set the compute interval in msec
           compute_interval_msec = (double) atof(optarg);
           break;
        case 'm':  // choose the comunication method : barrier, allreduce, alltoall, or exchange
           if (0 == strncasecmp(optarg, "barrier", 7))        method = BARRIER;
           else if (0 == strncasecmp(optarg, "alltoall", 8))  method = ALLTOALL;
           else if (0 == strncasecmp(optarg, "allreduce", 9)) method = ALLREDUCE;
           else if (0 == strncasecmp(optarg, "exchange", 8))  method = EXCHANGE;
           else                                               method = EXCHANGE;
           break;
        case 'k':  // choose a computation kernel
           if (0 == strncasecmp(optarg, "lut", 3))         kernel = 1;
           else if (0 == strncasecmp(optarg, "sqrt", 4))   kernel = 0;
           else                                            kernel = 0;
           break;
        case 't':  // set the target measurement time in sec
           target_measurement_time = atoi(optarg);
           break;
        case 'n':  // set the number of histogram bins per decade
           numbpd = atoi(optarg);
           break;
        case 'x':  // set the number of bytes for neighbor exchange or alltoall
           msgsize = atoi(optarg);
           break;
        case 'd':  // optionally dump all timing data
           dump_data = 1;
           break;
        default:
           break;
     }   
  }

  maxiter = (int) (1.0e3 * target_measurement_time / compute_interval_msec );

  if (help_flag) {
     if (myrank == 0) print_help();
     MPI_Finalize();
     return 0;
  }

  if (myrank == 0) {
     if (method == EXCHANGE) {
        printf("using neighbor exchange : compute_interval_msec = %.2lf, target measurement time = %.1lf sec, bins_per_decade = %d, msgsize = %d\n",
                compute_interval_msec, target_measurement_time, numbpd, msgsize);
     }
     else if (method == ALLTOALL) {
        printf("using alltoall on columns : compute_interval_msec = %.2lf, target measurement time = %.1lf sec, bins_per_decade = %d, msgsize = %d\n",
                compute_interval_msec, target_measurement_time, numbpd, msgsize);
     }
     else {
        printf("using global collectives : compute_interval_msec = %.2lf, target measurement time = %.1lf sec, bins_per_decade = %d\n",
                compute_interval_msec, target_measurement_time, numbpd);
     }
  }

  // choose a 2d decomposition
  npex = (int) (0.1 + sqrt((double) nranks));
  while (nranks % npex != 0) npex++;
  npey = nranks / npex;

  dblcount = msgsize / sizeof(double);

  mycol = myrank % npex;
  myrow = myrank / npex;

  if (method == ALLTOALL) {
     color = mycol;
     key = myrank;
     MPI_Comm_split(MPI_COMM_WORLD, color, key, &column_comm);
  }

  if (myrank == 0) printf("starting with nranks = %d\n", nranks);

  if (myrank == 0) { 
     if ( (method == EXCHANGE) || (method == ALLTOALL) )  printf("using npex = %d, npey = %d\n", npex, npey);
  }

  tcomm = (double *) malloc(maxiter*sizeof(double));
  tcomp = (double *) malloc(maxiter*sizeof(double));
  tstep = (double *) malloc(maxiter*sizeof(double));

  sendn = (double *) malloc(msgsize);
  sends = (double *) malloc(msgsize);
  sende = (double *) malloc(msgsize);
  sendw = (double *) malloc(msgsize);
  recvn = (double *) malloc(msgsize);
  recvs = (double *) malloc(msgsize);
  recve = (double *) malloc(msgsize);
  recvw = (double *) malloc(msgsize);

  for (i = 0; i< dblcount; i++) sendn[i] = (double) myrank;
  for (i = 0; i< dblcount; i++) sends[i] = (double) myrank;
  for (i = 0; i< dblcount; i++) sende[i] = (double) myrank;
  for (i = 0; i< dblcount; i++) sendw[i] = (double) myrank;

  for (i = 0; i< dblcount; i++) recvn[i] = 0.0;
  for (i = 0; i< dblcount; i++) recvs[i] = 0.0;
  for (i = 0; i< dblcount; i++) recve[i] = 0.0;
  for (i = 0; i< dblcount; i++) recvw[i] = 0.0;

  if (method == ALLTOALL) {
     sbuf = (double *) malloc(msgsize*npey);
     rbuf = (double *) malloc(msgsize*npey);
     for (i = 0; i< dblcount*npey; i++) sbuf[i] = (double) myrank;
     for (i = 0; i< dblcount*npey; i++) rbuf[i] = 0.0;
  }

  xrand = (double *) malloc(nrand*sizeof(double));

  srand48(13579L);
  for (i = 0; i < nrand; i++) xrand[i] = drand48();

  // initial guess for how many outer iterations of the compute kernel
  npts = (long) ( 2.0e7 * compute_interval_msec / 125.0 ); 
  ssum = 0.0;

  calib_iters = (int) ( 200.0 * 50.0 / compute_interval_msec );

  MPI_Barrier(MPI_COMM_WORLD);

  // make one call that is not timed
  compute(kernel, npts, nrand, xrand, &ssum);
  ssum += 1.0;

  if (method == EXCHANGE)       exch(0);
  else if (method == ALLTOALL)  MPI_Alltoall(sbuf, dblcount, MPI_DOUBLE, rbuf, dblcount, MPI_DOUBLE, column_comm);
  else if (method == ALLREDUCE) MPI_Allreduce(MPI_IN_PLACE, &ssum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  else                          MPI_Barrier(MPI_COMM_WORLD);

  if (myrank == 0) printf("calibrating compute time ..\n");

  MPI_Barrier(MPI_COMM_WORLD);

  // time calls to the compute routine with npts = 10^7 and adjust npts
  // compute for long enough to ramp up power
  t1 = MPI_Wtime();
  for (i = 0; i < calib_iters ; i++) {
    compute(kernel, npts, nrand, xrand, &ssum);
    ssum += 1.0;
  }
  t2 = MPI_Wtime();
  tmin = (t2 - t1)/((double) calib_iters);

  if (myrank == 0) printf("first-pass tmin = %.2lf msec\n", 1.0e3*tmin);

  // define a minimum time and use that as the reference
  // take the average-over-nodes of the node-local minimum time
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_DOUBLE, MPI_MIN, local_comm);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmin = tmin / ((double) nranks);

  // reset npts using measured compute time
  npts = (long) (((double) npts)*compute_interval_msec/(1.0e3*tmin));

  if (myrank == 0) printf("first iteration : npts = %ld\n", npts);

  // repeat this process one more time
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (i = 0; i < calib_iters ; i++) {
    compute(kernel, npts, nrand, xrand, &ssum);
    ssum += 1.0;
  }
  t2 = MPI_Wtime();
  tmin = (t2 - t1)/((double) calib_iters);

  // re-define tmin : average-over-nodes of the node-local minimum time
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_DOUBLE, MPI_MIN, local_comm);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmin = tmin / ((double) nranks);

  // reset npts using measured compute time
  npts = (long) (((double) npts)*compute_interval_msec/(1.0e3*tmin));

  if (myrank == 0) printf("using npts = %ld, tmin = %.2lf msec\n\n", npts, 1.0e3*tmin);

  compute_interval_msec = 1.0e3*tmin;

  getrusage(RUSAGE_SELF, &RU);
  context_switches_initial = RU.ru_nivcsw;
  usr_cpu_initial          = RU.ru_utime.tv_sec + 1.0e-6*RU.ru_utime.tv_usec;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Pcontrol(1);
  
  elapsed1 = MPI_Wtime();

  //==============================================
  // time a number of {compute, communicate} steps
  //==============================================
  for (iter = 0; iter < maxiter; iter++) {
    if ( barrier_flag && ((iter + 1) % 100 == 0) ) MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    compute(kernel, npts, nrand, xrand, &ssum);
    ssum += 1.0;
    t2 = MPI_Wtime();
    if (method == EXCHANGE)       exch(iter);
    else if (method == ALLTOALL)  MPI_Alltoall(sbuf, dblcount, MPI_DOUBLE, rbuf, dblcount, MPI_DOUBLE, column_comm);
    else if (method == ALLREDUCE) MPI_Allreduce(MPI_IN_PLACE, &ssum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    else                          MPI_Barrier(MPI_COMM_WORLD);
    t3 = MPI_Wtime();
    tcomp[iter] = t2 - t1;
    tcomm[iter] = t3 - t2;
    tstep[iter] = t3 - t1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  elapsed2 = MPI_Wtime();

  MPI_Pcontrol(0);

  getrusage(RUSAGE_SELF, &RU);
  context_switches = RU.ru_nivcsw - context_switches_initial;
  usr_cpu_final    = RU.ru_utime.tv_sec + 1.0e-6*RU.ru_utime.tv_usec;
  usr_time = usr_cpu_final - usr_cpu_initial;

  sum_comm = 0.0;
  for (iter = 0; iter < maxiter; iter++) sum_comm += tcomm[iter];

  myMPI.value = sum_comm;
  myMPI.index = myrank;
  MPI_Allreduce(&myMPI, &minMPI, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_comm, &avgMPI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avgMPI = avgMPI / ((double) nranks);

  // after this call, tcomm holds the max time in MPI over all ranks for each step
  MPI_Allreduce(MPI_IN_PLACE, tcomm, maxiter, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // compute mean and relative variation rank by rank
  mean_comp = 0.0;
  for (iter = 0; iter < maxiter; iter++) mean_comp += tcomp[iter];
  mean_comp = mean_comp / ((double) maxiter);

  ssq = 0.0;
  for (iter = 0; iter < maxiter; iter++) ssq += (tcomp[iter] - mean_comp) * (tcomp[iter] - mean_comp);

  sigma = sqrt(ssq/((double) maxiter));

  relative_variation = 100.0*sigma/mean_comp;

  alldev_comp = (double *) malloc(nranks*sizeof(double));
  allavg_comp = (double *) malloc(nranks*sizeof(double));
  allsum_comm = (double *) malloc(nranks*sizeof(double));
  allusr_time = (double *) malloc(nranks*sizeof(double));
  all_context_switches = (long *) malloc(nranks*sizeof(long));
  all_cpus = (int *) malloc(nranks*sizeof(int));

  mycpu = sched_getcpu();

  MPI_Gather(&relative_variation, 1, MPI_DOUBLE, alldev_comp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&mean_comp,          1, MPI_DOUBLE, allavg_comp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&sum_comm,           1, MPI_DOUBLE, allsum_comm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&usr_time,           1, MPI_DOUBLE, allusr_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&context_switches,   1, MPI_LONG,   all_context_switches, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&mycpu,              1, MPI_INT,    all_cpus, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // data volume per node (most of this is likely to stay on-node)
  if (npex == 1) xtraffic = 0.0;
  else           xtraffic = (double) ( (4*(npex - 2) + 2)*npey ) * ((double) msgsize);
  if (npey == 1) ytraffic = 0.0;
  else           ytraffic = (double) ( (4*(npey - 2) + 2)*npex ) * ((double) msgsize);
  data_volume = (xtraffic + ytraffic) / ((double) num_nodes);

  tmax = 0.0; tmin = 1.0e30;
  for (iter = 0; iter < maxiter; iter++) {
    if (tcomm[iter] > tmax) tmax = tcomm[iter];
    if (tcomm[iter] < tmin) tmin = tcomm[iter];
  }

  if (myrank == 0) printf("max communication time in msec = %.3lf\n",  1.0e3*tmax);

  // use log-scale binning
  log10_tmin = log10(tmin);
  log10_tmax = log10(tmax);
  
  dnumbpd = (double) numbpd;
  botp = floor(log10_tmin);
  botn = floor(dnumbpd*(log10_tmin - botp));
  topp = floor(log10_tmax);
  topn = ceil(dnumbpd*(log10_tmax - topp));

  // total number of histogram bins
  totbins = (int) round( (dnumbpd*topp + topn) - (dnumbpd*botp + botn) );

  histo = (long *) malloc(totbins*sizeof(long));
   
  for (bin = 0; bin < totbins; bin++) histo[bin] = 0L;

  for (iter = 0; iter < maxiter; iter++) {
    bin = (int) ( dnumbpd*log10(tcomm[iter]) - (dnumbpd*botp + botn) );
    if ((bin >= 0) && (bin < totbins)) histo[bin]++;
  }

  if (myrank == 0) {
    printf("\n");
    printf("histogram of max communication times per step, in units of msec\n");
    printf(" [     min -        max ):      count\n");
    for (bin = 0; bin < totbins; bin++) {
      exp1 = botp + (botn + ((double) bin)) / dnumbpd;
      exp2 = exp1 + 1.0 / dnumbpd;
      hmin = 1.0e3*pow(10.0, exp1);
      hmax = 1.0e3*pow(10.0, exp2);
      printf("%10.3lf - %10.3lf  : %10ld \n", hmin, hmax, histo[bin]);
    }
  }

  free(histo);

  // for per-node analysis, focus on the compute times
  compmax = (double *) malloc(maxiter*sizeof(double));

  // find the global minimum computation time over all iterations
  compmin = 1.0e30;
  for (iter = 0; iter < maxiter; iter++) {
    if (tcomp[iter] < compmin) compmin = tcomp[iter];
  }
  
  MPI_Allreduce(MPI_IN_PLACE, &compmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  // find the max compute time for each iteration on a per-node basis
  MPI_Allreduce(tcomp, compmax, maxiter, MPI_DOUBLE, MPI_MAX, local_comm);

  // find the global avg time for computation
  compavg = 0.0;
  for (iter = 0; iter < maxiter; iter++) compavg += tcomp[iter];

  MPI_Allreduce(MPI_IN_PLACE, &compavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  samples_total = ((double) maxiter) * ((double) nranks);

  compavg = compavg / samples_total;

  // look at the max computation times per iteration for analysis by node
  compssq = 0.0;
  for (iter = 0; iter < maxiter; iter++) {
    dev = compmax[iter] - compavg;
    compssq += dev*dev;
  }

  sigma_comp = sqrt(compssq/((double) maxiter));

  gethostname(host, sizeof(host));

  for (i=0; i<sizeof(host); i++) {    
     if (host[i] == '.') {
        host[i] = '\0';
        break;
     }
  }

  hostlen = strlen(host);
  MPI_Allreduce(&hostlen, &maxlen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (local_rank == 0) {

     aggregate_sigma_comp = (double *) malloc(num_nodes*sizeof(double));
     aggregate_compavg    = (double *) malloc(num_nodes*sizeof(double));

     MPI_Allgather(&sigma_comp, 1, MPI_DOUBLE, aggregate_sigma_comp, 1, MPI_DOUBLE, node_comm);
     MPI_Allgather(&compavg,    1, MPI_DOUBLE, aggregate_compavg,    1, MPI_DOUBLE, node_comm);

     snames = (char *) malloc(num_nodes*sizeof(host));
     rnames = (char *) malloc(num_nodes*sizeof(host));
     sort_key = (int *) malloc(num_nodes*sizeof(int));

     for (i=0; i<num_nodes; i++)  {
       ptr = snames + i*sizeof(host);
       strncpy(ptr, host, sizeof(host));
     }

     MPI_Alltoall(snames, sizeof(host), MPI_BYTE, rnames, sizeof(host), MPI_BYTE, node_comm);

     // sort by node-local sigma for computation times
     // aggregate_sigma_comp is returned in sorted order, along with the sort key
     sortx(aggregate_sigma_comp, num_nodes, sort_key, SORT_DESCENDING_ORDER);

     sprintf(hfmt, "%%%ds", maxlen);
     sprintf(heading, hfmt, "host");
     strcat(heading, "         mean(msec)    percent variation\n");
     printf(heading);
     sprintf(format, "%%%ds    %10.3lf    %10.3lf", maxlen);
     strcat(format, "\n");

     if (myrank == 0) {
        printf("\n");
        printf(" percent variation = 100*sigma/mean for the max computation times per step by node:\n");
        printf(heading);
        for (i=0; i< num_nodes; i++) {
           k = sort_key[i];
           ptr = rnames + k*sizeof(host);
           relative_variation = 100.0*aggregate_sigma_comp[i] / aggregate_compavg[k];
           printf(format, ptr, 1.0e3*aggregate_compavg[k], relative_variation);
        }
        printf("\n");
     }
  }

  memset(hfmt, '\0', sizeof(hfmt));
  memset(heading, '\0', sizeof(heading));
  memset(format, '\0', sizeof(format));
  

  // analyze all step time data for all ranks
  tmin = 1.0e30; tmax = 0.0;
  for (iter = 0; iter < maxiter; iter++) {
     if (tstep[iter] > tmax) tmax = tstep[iter];
     if (tstep[iter] < tmin) tmin = tstep[iter];
  }

  if ( (method == ALLREDUCE) || (method == BARRIER) ) {
     // in this case there is just one step time per iteration, the same for all ranks
     tavg = (elapsed2 - elapsed1) / ((double) maxiter);
     
     // compute moments for the time-step samples
     ssq = 0.0; scb = 0.0; sp4 = 0.0; 
     for (iter = 0; iter < maxiter; iter++) {
       dev = tstep[iter] - tavg;
       ssq += dev*dev;
       scb += dev*dev*dev;
       sp4 += dev*dev*dev*dev;
     }

     samples = ((double) maxiter);
  }
  else {
     // analyze all step time data for every rank
     tavg = 0.0;
     for (iter = 0; iter < maxiter; iter++) tavg += tstep[iter];

     // global avg for step times
     MPI_Allreduce(MPI_IN_PLACE, &tavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
     samples = ((double) maxiter) * ((double) nranks);
    
     tavg = tavg / samples;

     // compute moments for the time-step samples ... per rank
     ssq = 0.0; scb = 0.0; sp4 = 0.0; 
     for (iter = 0; iter < maxiter; iter++) {
       dev = tstep[iter] - tavg;
       ssq += dev*dev;
       scb += dev*dev*dev;
       sp4 += dev*dev*dev*dev;
     }

     // compute global statistics on the step times
     MPI_Allreduce(MPI_IN_PLACE, &ssq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(MPI_IN_PLACE, &scb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(MPI_IN_PLACE, &sp4, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  sigma = sqrt(ssq/samples);

  relative_variation = sigma / tavg;

  skewness = scb / (samples * sigma * sigma * sigma);

  kurtosis = sp4 / (samples * sigma * sigma * sigma * sigma);

  // find the global min and max step times
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // use log-scale binning
  log10_tmin = log10(tmin);
  log10_tmax = log10(tmax);
  
  dnumbpd = (double) numbpd;
  botp = floor(log10_tmin);
  botn = floor(dnumbpd*(log10_tmin - botp));
  topp = floor(log10_tmax);
  topn = ceil(dnumbpd*(log10_tmax - topp));

  // total number of histogram bins
  totbins = (int) round( (dnumbpd*topp + topn) - (dnumbpd*botp + botn) );

  histo = (long *) malloc(totbins*sizeof(long));
   
  for (bin = 0; bin < totbins; bin++) histo[bin] = 0L;

  // each rank histograms its own data
  for (iter = 0; iter < maxiter; iter++) {
    bin = (int) ( dnumbpd*log10(tstep[iter]) - (dnumbpd*botp + botn) );
    if ((bin >= 0) && (bin < totbins)) histo[bin]++;
  }

  // for the exchange or alltoall methods, we should sum over all MPI ranks
  if ( (method == EXCHANGE) || (method == ALLTOALL) ) {
     MPI_Allreduce(MPI_IN_PLACE, histo, totbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  }

  if (myrank == 0) {
    printf("histogram of step times in msec for all ranks\n");
    printf(" [     min -        max ):      count\n");
    for (bin = 0; bin < totbins; bin++) {
      exp1 = botp + (botn + ((double) bin)) / dnumbpd;
      exp2 = exp1 + 1.0 / dnumbpd;
      hmin = 1.0e3*pow(10.0, exp1);
      hmax = 1.0e3*pow(10.0, exp2);
      printf("%10.3lf - %10.3lf  : %10ld \n", hmin, hmax, histo[bin]);
    }
    printf("\n");
  }

  free(histo);

  // optionally dump data in a single binary file, use floats to save space
  if (dump_data) {
    floatcomp = (float *) malloc(maxiter*sizeof(float));
    floatstep = (float *) malloc(maxiter*sizeof(float));
    for (iter=0; iter<maxiter; iter++) {
      floatcomp[iter] = (float) tcomp[iter];
      floatstep[iter] = (float) tstep[iter];
    }
    if (myrank == 0) {
      compbuf = (float *) malloc(maxiter*sizeof(float));
      stepbuf = (float *) malloc(maxiter*sizeof(float));
      ptr = getenv("LSB_JOBID");
      if (ptr == NULL) jobid = getpid();
      else             jobid = atoi(ptr);
      sprintf(outfile, "%d.timing_data", jobid);
      ofp = fopen(outfile, "w");
      if (ofp != NULL) {
        fwrite(&nranks, sizeof(int), 1, ofp);
        fwrite(&maxiter, sizeof(int), 1, ofp);
        fwrite(floatcomp, sizeof(float), maxiter, ofp);
        fwrite(floatstep, sizeof(float), maxiter, ofp);
      }
      for (rank=1; rank<nranks; rank++) {
         MPI_Send(&green_light, 1, MPI_INT, rank, tag, MPI_COMM_WORLD);
         MPI_Recv(compbuf, maxiter, MPI_FLOAT, rank, tag, MPI_COMM_WORLD, &status);
         MPI_Recv(stepbuf, maxiter, MPI_FLOAT, rank, tag, MPI_COMM_WORLD, &status);
         if (ofp != NULL) {
           fwrite(compbuf, sizeof(float), maxiter, ofp);
           fwrite(stepbuf, sizeof(float), maxiter, ofp);
         }
      }
    }
    else {
      MPI_Recv(&green_light, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
      MPI_Send(floatcomp, maxiter, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
      MPI_Send(floatstep, maxiter, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }
  }

  if (myrank == 0) {
    bw = 1.0e-9*data_volume/(tavg - compmin);
    printf("min computational interval in msec = %.4lf\n", 1.0e3*compmin);
    printf("the average time per step in msec  = %.4lf\n", 1.0e3*tavg);
    printf("percent variation in step times    = %.3lf\n", 100.0*relative_variation);
    printf("skewness of step time distribution = %.3le\n", skewness);
    printf("kurtosis of step time distribution = %.3le\n", kurtosis);
    printf("elapsed time (sec)                 = %.3lf\n", elapsed2 - elapsed1);
    printf("measured time / target time        = %.3lf\n", (elapsed2 - elapsed1)/target_measurement_time); 
    printf("effective parallel efficiency      = %.3lf\n", target_measurement_time/(elapsed2 - elapsed1));
    if (method == EXCHANGE) printf("effective exch bw per node         = %.3le GB/sec\n", bw);
    printf("ssum =  %.6le\n\n", ssum);
  }

  if (myrank == 0) {
    maxavg = 0.0;
    minavg = 1.0e30;
    maxrank = 0;
    minrank = 0;
    for (i=0; i<nranks; i++) {
      if (allavg_comp[i] > maxavg)  {
        maxavg = allavg_comp[i];
        maxrank = i;
      }   
      if (allavg_comp[i] < minavg)  {
        minavg = allavg_comp[i];
        minrank = i;
      }   
    }   
    printf("\n");
    printf("max avg compute interval = %.3lf msec for rank %d\n", 1.0e3*maxavg, maxrank);
    printf("min avg compute interval = %.3lf msec for rank %d\n", 1.0e3*minavg, minrank);
    printf("min MPI time = %.3lf sec for rank %d\n", minMPI.value, minMPI.index);
    printf("avg MPI time = %.3lf sec\n", avgMPI);
    printf("\n");
    printf("avg compute interval (msec), percent relative variation of compute times, and total communication time (sec) by rank:\n");
    sprintf(hfmt, "%%s %%%ds", maxlen);
    sprintf(heading, hfmt, "  rank", "host");
    strcat(heading, "    cpu  avg_comp(msec)    %%reldev   total_comm(sec)   user_time(sec)   switches\n");
    printf(heading);
    sprintf(format, "%%6d %%%ds %%6d %%14.3lf  %%10.2lf %%16.3lf %%16.3lf %%11ld", maxlen);
    strcat(format, "\n");
    for (i=0; i<nranks; i++) {
       k = i / ranks_per_node;
       ptr = rnames + k*sizeof(host);
       printf(format, i, ptr, all_cpus[i], 1.0e3*allavg_comp[i], alldev_comp[i], allsum_comm[i], allusr_time[i], all_context_switches[i]);
    }
    printf("\n");
  }

  MPI_Finalize();

  return 0;
}


// -----------------------------------
// routine for boundary exchange
// -----------------------------------
void exch(int iter)
{
  int ireq, tag, north, south, east, west;
  MPI_Request req[8];
  MPI_Status status[8];

  ireq = 0;

  tag = 99 + 20*iter;

  // exchange data with partner to the north
  if (myrow != (npey - 1)) {
    north = myrank + npex;
    MPI_Irecv(recvn, dblcount, MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
    MPI_Isend(sendn, dblcount, MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
  }

  // exchange data with partner to the east
  if (mycol != (npex - 1)) {
    east = myrank + 1;
    MPI_Irecv(recve, dblcount, MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
    MPI_Isend(sende, dblcount, MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
  }

  // exchange data with partner to the south
  if (myrow != 0) {
    south = myrank - npex;
    MPI_Irecv(recvs, dblcount, MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
    MPI_Isend(sends, dblcount, MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
  }

  // exchange data with partner to the west
  if (mycol != 0) {
    west = myrank - 1;
    MPI_Irecv(recvw, dblcount, MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
    MPI_Isend(sendw, dblcount, MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &req[ireq]);
    ireq = ireq + 1;
  }

  MPI_Waitall(ireq, req, status);

  return;
  
}

// -----------------------------------
// routine for computation
// -----------------------------------
void compute(int flag, long n, int nrand, double * xrand, double * ssum)
{
  double tsum;
  long i, lrand, ndx;

  lrand = (long) nrand;
  
  tsum = 0.0;

  if (flag) {
    for (i = 0L; i < n; i++) {
      ndx = i % lrand;
      tsum = tsum + lutexp(xrand[ndx]);
    }
  }
  else {
    for (i = 0L; i < n; i++) {
      ndx = i % lrand;
      tsum = tsum + sqrt(1.0 + xrand[ndx]);
    }
  }

  *ssum = tsum;

}

// -----------------------------------
// help message
// -----------------------------------
void print_help(void)
{
   printf("Syntax: mpirun -np #ranks osnoise [-c compute_interval_msec] [-t target_measurement_time] [-n histogram_bins_per_decade] [-x msgsize] [-m method] [-k kernel] [-d] [-b]\n");
   printf(" -c float ... specifies the compute interval in milliseconds\n");
   printf(" -t int   ... specifies the target measurement time in units of seconds\n");
   printf(" -n int   ... specifies the number of histogram bins per decade\n");
   printf(" -x int   ... specifies the message size in bytes used for neighbor exchange or alltoall\n");
   printf(" -m char  ... specifies the communication method (values : exchange, alltoall, allreduce, barrier)\n");
   printf(" -k char  ... specifies the compute kernel (values : sqrt, lut)\n");
   printf(" -d       ... sets flag to dump data to a file\n");
   printf(" -b       ... sets flag to add a barrier every 100 iterations of the (compute, communicate) loop\n");    
}


//===========================================================================
// incremental Shell sort with increment array: inc[k] = 1 + 3*2^k + 4^(k+1) 
//===========================================================================
void sortx(double * arr , int n, int * ind, int flag)
{
   int h, i, j, k, inc[20];
   int numinc, pwr2, pwr4;
   double val;

   if (n <= 1) {
      ind[0] = 0;
      return;
   }

   pwr2 = 1;
   pwr4 = 4;

   numinc = 0;
   h = 1;
   inc[numinc] = h;
   while (numinc < 20) {
      h = 1 + 3*pwr2 + pwr4;
      if (h > n) break;
      numinc++;
      inc[numinc] = h;
      pwr2 *= 2;
      pwr4 *= 4;
   }

   for (i=0; i<n; i++) ind[i] = i;

   if (flag > 0) { // sort in increasing order 
      for (; numinc >= 0; numinc--) {
         h = inc[numinc];
         for (i = h; i < n; i++) {
            val = arr[i];
            k   = ind[i];

            j = i;
   
            while ( (j >= h) && (arr[j-h] > val) ) {
               arr[j] = arr[j-h];
               ind[j] = ind[j-h];
               j = j - h;
            }

            arr[j] = val;
            ind[j] = k;
         }
      }
   }
   else { // sort in decreasing order 
      for (; numinc >= 0; numinc--) {
         h = inc[numinc];
         for (i = h; i < n; i++) {
            val = arr[i];
            k   = ind[i];

            j = i;
   
            while ( (j >= h) && (arr[j-h] < val) ) {
               arr[j] = arr[j-h];
               ind[j] = ind[j-h];
               j = j - h;
            }

            arr[j] = val;
            ind[j] = k;
         }
      }
   }
}
