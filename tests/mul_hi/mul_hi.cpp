#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "mul_hi.hpp"

/* Test result of mul_hi on C++
@params X = 64-bit unsigned integer
@params Y = 64-bit unsigned integer
@params Z = 128-bit unsigned integer (result of X * Y)
@params Z_hi = 64-bit unsigned integer (most significant 64-bits of Z)
*/

#define X2 0xdead
#define Y2 0xbeef
#define X1 0xffffffff
#define Y1 0xdeadbeef
#define X0 0xdeadbeefdeadbeef
#define Y0 0xffffffffffffffff

typedef __uint128_t uint128_t;

int main(int argc, char **argv) {
    std::cout << "Working on uint16 numbers:" << std::endl;
    uint16_t a0 = (uint16_t)(X2);
    uint16_t b0 = (uint16_t)(Y2);
    uint16_t c0 = mul_hi16(a0, b0);
    std::cout << "Result: " << std::hex << c0 << std::endl;
    uint32_t d0 = ((uint32_t)(a0)) * ((uint32_t)(b0)); // Work in 32-bits to get result
    uint16_t d_hi0 = (uint16_t)(d0 >> 16);
    std::cout << "Actual: " << std::hex << d_hi0 << std::endl;

    std::cout << std::endl << "Working on uint32 numbers:" << std::endl;
    uint32_t a1 = (uint32_t)(X1);
    uint32_t b1 = (uint32_t)(Y1);
    uint32_t c1 = mul_hi32(a1, b1);
    std::cout << "Result: " << std::hex << c1 << std::endl;
    uint64_t d1 = ((uint64_t)(a1)) * ((uint64_t)(b1)); // Work in 64-bits to get result
    uint32_t d_hi1 = (uint32_t)(d1 >> 32);
    std::cout << "Actual: " << std::hex << d_hi1 << std::endl;

    std::cout << std::endl << "Initializing uint64 numbers:" << std::endl;
    uint64_t x_lo = X0 & 0x00000000ffffffff;
    uint64_t x_hi = X0 >> 32;
    uint64_t y_lo = Y0 & 0x00000000ffffffff;
    uint64_t y_hi = Y0 >> 32;
    std::cout << "x_hi : " << std::hex << x_hi << ", x_lo : " << std::hex << x_lo << std::endl;
    std::cout << "y_hi : " << std::hex << y_hi << ", y_lo : " << std::hex << y_lo << std::endl;

    std::cout << "Calculating partial products..." << std::endl;
    uint64_t x_x_y_lo = x_lo * y_lo;
    uint64_t x_x_y_mid = x_hi * y_lo;
    uint64_t y_x_x_mid = y_hi * x_lo;
    uint64_t x_x_y_hi = x_hi * y_hi;
    uint64_t carry_bit = (x_x_y_mid & 0x00000000ffffffff) +
                         (y_x_x_mid & 0x00000000ffffffff) +
                         (x_x_y_lo >> 32);
    carry_bit >>= 32;
    std::cout << "Carry bit: " << std::hex << carry_bit << std::endl;

    uint64_t multhi = x_x_y_hi +
                      (x_x_y_mid >> 32) + (y_x_x_mid >> 32) +
                      carry_bit;

    std::cout << "Result: " << std::hex << multhi << std::endl;

    std::cout << "Alt V : " << std::hex << mul_hi64(X0, Y0) << std::endl;
    std::cout << "Attempt 128-bit calculations..." << std::endl;

    uint128_t x0 = (uint128_t)(X0);
    uint128_t y0 = (uint128_t)(Y0);
    uint128_t res = x0*y0;
    uint64_t hiBits = (uint64_t)(res >> 64);
    std::cout << "Result: " << std::hex << hiBits <<std::endl;
    std::cout << "\nComplete..." << std::endl;

    return 0;
}
