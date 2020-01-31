#include <stdint.h>
#include <stdlib.h>

/*
  Returns most significant 64-bit of result of
  multiplication between two 64-bit numbers
*/
uint64_t mul_hi64(const uint64_t x, const uint64_t y) {
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

/*
  Returns most significant 32-bit of result of
  multiplication between two 32-bit numbers
*/
uint32_t mul_hi32(const uint32_t x, const uint32_t y) {
    uint32_t x_lo = x & 0x0000ffff;
    uint32_t x_hi = x >> 16;
    uint32_t y_lo = y & 0x0000ffff;
    uint32_t y_hi = y >> 16;
    uint32_t x_x_y_lo = x_lo * y_lo;
    uint32_t x_x_y_mid = x_hi * y_lo;
    uint32_t y_x_x_mid = y_hi * x_lo;
    uint32_t x_x_y_hi = x_hi * y_hi;
    uint32_t carry_bit = (x_x_y_mid & 0x0000ffff) +
                         (y_x_x_mid & 0x0000ffff) +
                         (x_x_y_lo >> 16);
    carry_bit >>= 16;
    uint64_t multhi = x_x_y_hi +
                      (x_x_y_mid >> 16) + (y_x_x_mid >> 16) +
                      carry_bit;
    return multhi;
}

/*
  Returns most significant 16-bit of result of
  multiplication between two 16-bit numbers
*/
uint16_t mul_hi16(const uint16_t x, const uint16_t y) {
    uint16_t x_lo = x & 0x00ff;
    uint16_t x_hi = x >> 8;
    uint16_t y_lo = y & 0x00ff;
    uint16_t y_hi = y >> 8;
    uint16_t x_x_y_lo = x_lo * y_lo;
    uint16_t x_x_y_mid = x_hi * y_lo;
    uint16_t y_x_x_mid = y_hi * x_lo;
    uint16_t x_x_y_hi = x_hi * y_hi;
    uint16_t carry_bit = (x_x_y_mid & 0x00ff) +
                         (y_x_x_mid & 0x00ff) +
                         (x_x_y_lo >> 8);
    carry_bit >>= 8;
    uint16_t multhi = x_x_y_hi +
                      (x_x_y_mid >> 8) + (y_x_x_mid >> 8) +
                      carry_bit;
    return multhi;
}

