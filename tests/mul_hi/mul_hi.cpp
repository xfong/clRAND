#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

/* Test result of mul_hi on C++
@params X = 64-bit unsigned integer
@params Y = 64-bit unsigned integer
@params Z = 128-bit unsigned integer (result of X * Y)
@params Z_hi = 64-bit unsigned integer (most significant 64-bits of Z)
*/

#define X0 0xdeadbeefdeadbeef
#define Y0 0xffffffffffffffff

typedef __uint128_t uint128_t;

uint64_t mul_hi(const uint64_t x, const uint64_t y) {
    uint64_t x_lo = x & 0x00000000ffffffff;
    uint64_t x_hi = x >> 32;
    uint64_t y_lo = y & 0x00000000ffffffff;
    uint64_t y_hi = y >> 32;
    uint64_t x_x_y_lo = x_lo * y_lo;
    uint64_t x_x_y_mid = x_hi * y_lo;
    uint64_t y_x_x_mid = y_hi * x_lo;
    uint64_t x_x_y_hi = x_hi * y_hi;
    uint64_t carry_bit = (x_x_y_mid & 0x00000000ffffffff) +
                         (y_x_x_mid & 0x00000000ffffffff) +
                         (x_x_y_lo >> 32);
    carry_bit >>= 32;
    uint64_t multhi = x_x_y_hi +
                      (x_x_y_mid >> 32) + (y_x_x_mid >> 32) +
                      carry_bit;
    return multhi;
}

int main(int argc, char **argv) {
    std::cout << "Initializing numbers:" << std::endl;
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

    std::cout << "Alt V : " << std::hex << mul_hi(X0, Y0) << std::endl;
    std::cout << "Attempt 128-bit calculations..." << std::endl;

    uint128_t x0 = (uint128_t)(X0);
    uint128_t y0 = (uint128_t)(Y0);
    uint128_t res = x0*y0;
    uint64_t hiBits = (uint64_t)(res >> 64);
    std::cout << "Result: " << std::hex << hiBits <<std::endl;
    std::cout << "\nComplete..." << std::endl;

    return 0;
}
