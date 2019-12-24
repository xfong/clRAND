# clPRNG
C++ library of pseudo-random number generators for OpenCL devices

The PRNG is saved in a data structure which contains the command
queue it will execute on, the program and kernels, and temporary
buffers. The random bitstreams are created in the temporary
buffer store and tracked. They are then copied from the
temporary store to the desired destination. If the temporary
store becomes empty, we will generate a new bitstream to
replenish it fully. The storage in the temporary store should
accomodate two random numbers from every workitem that will run

TODO:
1) Need functions to build the program for the PRNG
    a) One function needs to build up the text source that is
       stored in the data structure. A flag needs to be set
       to say the source is ready.
    b) One function takes the text source and builds the
       program
    c) One function builds two kernels from the program. One
       kernel is to initialize the PRNG state. The other kernel
       is called to generate the random bitstream

