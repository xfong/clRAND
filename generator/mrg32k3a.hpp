typedef struct {
   unsigned int g1[3];
   unsigned int g2[3];
} mrg32k3a_state;

const char * mrg32k3a_prng_kernel = R"EOK(
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Thomas Bradley, Parallelisation Techniques for Random Number Generators
// https://www.nag.co.uk/IndustryArticles/gpu_gems_article.pdf

#define MRG32K3A_POW32 4294967296
#define MRG32K3A_M1 4294967087
#define MRG32K3A_M1C 209
#define MRG32K3A_M2 4294944443
#define MRG32K3A_M2C 22853
#define MRG32K3A_A12 1403580
#define MRG32K3A_A13 (4294967087 -  810728)
#define MRG32K3A_A13N 810728
#define MRG32K3A_A21 527612
#define MRG32K3A_A23 (4294944443 - 1370589)
#define MRG32K3A_A23N 1370589
#define MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/MRG32K3A_M1
#define MRG32K3A_UINT_NORM (1.000000048661607) // (MRG32K3A_POW32 - 1)/(MRG32K3A_M1 - 1)

/**
State of mrg32k3a RNG.
*/
typedef struct {
   unsigned int g1[3];
   unsigned int g2[3];
} mrg32k3a_state;

inline static
unsigned long long mad_u64_u32(const unsigned int x, const unsigned int y, const unsigned long long z)
{
    unsigned long long x0;
    unsigned long long y0;
    x0 = (unsigned long long)(x);
    y0 = (unsigned long long)(y);
    
    return (x0*y0) + z;
}

inline static
unsigned long long mod_m1(unsigned long long p){
    p = mad_u64_u32(MRG32K3A_M1C, (p >> 32), p & (MRG32K3A_POW32 - 1));
    if (p >= MRG32K3A_M1)
        p -= MRG32K3A_M1;

    return p;
}

inline static
unsigned long long mod_m2(unsigned long long p){
    p = mad_u64_u32(MRG32K3A_M2C, (p >> 32), p & (MRG32K3A_POW32 - 1));
    p = mad_u64_u32(MRG32K3A_M2C, (p >> 32), p & (MRG32K3A_POW32 - 1));
    if (p >= MRG32K3A_M2) {
        p -= MRG32K3A_M2;
    }

    return p;
}

inline static
unsigned long long mod_mul_m1(unsigned int i,
                              unsigned long long j){
    long long hi, lo, temp1, temp2;

    hi = i / 131072;
    lo = i - (hi * 131072);
    temp1 = mod_m1(hi * j) * 131072;
    temp2 = mod_m1(lo * j);
    lo = mod_m1(temp1 + temp2);

    if (lo < 0)
        lo += MRG32K3A_M1;
    return lo;
}

inline static
unsigned long long mod_mul_m2(unsigned int i,
                              unsigned long long j){
    long long hi, lo, temp1, temp2;

    hi = i / 131072;
    lo = i - (hi * 131072);
    temp1 = mod_m2(hi * j) * 131072;
    temp2 = mod_m2(lo * j);
    lo = mod_m2(temp1 + temp2);

    if (lo < 0)
        lo += MRG32K3A_M2;
    return lo;
}

/**
Generates a random 32-bit unsigned integer using mrg32k3a RNG.

@param state State of the RNG to use.
*/
#define mrg32k3a_uint(state) _mrg32k3a_uint(&state)
uint _mrg32k3a_uint(mrg32k3a_state* state){
    const unsigned int p1 = (unsigned int)mod_m1(
        mad_u64_u32(
            MRG32K3A_A12,
            state.g1[1],
            mad_u64_u32(
                MRG32K3A_A13N,
                (MRG32K3A_M1 - state.g1[0]),
                0
            )
        )
    );

    state.g1[0] = state.g1[1];
    state.g1[1] = state.g1[2];
    state.g1[2] = p1;

    const unsigned int p2 = (unsigned int)mod_m2(
        mad_u64_u32(
            MRG32K3A_A21,
            state.g2[2],
            mad_u64_u32(
                MRG32K3A_A23N,
                (MRG32K3A_M2 - state.g2[0]),
                0
            )
        )
    );

    state.g2[0] = state.g2[1];
    state.g2[1] = state.g2[2];
    state.g2[2] = p2;

    return (p1 - p2) + (p1 <= p2 ? MRG32K3A_M1 : 0);
}

/**
Seeds mrg31k3p RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mrg32k3a_seed(mrg32k3a_state* state, ulong j){
    unsigned int x = (unsigned int) j ^ 0x55555555U;
    unsigned int y = (unsigned int) ((j >> 32) ^ 0xAAAAAAAAU);
    state.g1[0] = mod_mul_m1(x, j);
    state.g1[1] = mod_mul_m1(y, j);
    state.g1[2] = mod_mul_m1(x, j);
    state.g2[0] = mod_mul_m2(y, j);
    state.g2[1] = mod_mul_m2(x, j);
    state.g2[2] = mod_mul_m2(y, j);
}
)EOK";
