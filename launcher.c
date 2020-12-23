/* Copyright IBM Corporation, 2020
 * author : Bob Walkup
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/personality.h>

int main(int argc, char * argv[])
{
  int rc;

  rc = personality(ADDR_NO_RANDOMIZE);
  if (rc == -1) fprintf(stderr, "personality failed ... not disabling ASLR\n");

  rc = execvp(argv[1], &argv[1]);
  if (rc == -1) {
    fprintf(stderr, "execvp failed for %s ... exiting\n", argv[1]);
    exit(0);
  }
  return 0;
}
