# clRAND
C++ library of pseudo-random number generators for OpenCL devices

The PRNG is saved in a data structure which contains the command
queue it will execute on, the program and kernels, and temporary
buffers. The random bitstreams are created in the temporary
buffer store and tracked. They are then copied from the
temporary store to the desired destination. If the temporary
store becomes empty, we will generate a new bitstream to
replenish it fully. The storage in the temporary store should
accomodate two random numbers from every workitem that will run

REQUIREMENTS:
- GCC v5+
- OpenCL 1.2

TODO:
1) Need functions to test the various PRNGs (Philox4x32_10,
   xorshift1024, Threefry, Sobol32)

2) Need function to generate the random initial seeds for PRNGs

3) Need functions to generate floats and double that are
   uniformly distributed in [0, 1).
    a) Can be generated quickly with some sacrifice in
       number of possible values (intrinsic functions)
    b) Can be generated accurately by using more uint or ulong
       random numbers to determine the exponent of the number
       (extrinsic functions)

4) Need functions to use floats and doubles that are
   uniformly distributed in [0, 1) to:
    a) Generate normally distributed random numbers
    b) Generate log-normally distributed random numbers
    b) Generate Poisson distributed random numbers

5) To add Mersenne Twister PRNGs for Mersenne prime 11213, 23209 and 44497
