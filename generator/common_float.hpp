const char * flt_kernel_util_defs = R"EOK(
#define float_inv_max_uint 2.3283064370807973754314699618685e-10‬f
#define float_inv_nmax_uint 2.3283064365386962890625‬e-10f

#define float_inv_max_ulong 5.4210108624275221703311375920553e-20‬f
#define float_inv_nmax_ulong 5.4210108624275221700372640043497e-20f

#define float_inv_nmax_uint_adj 1.16415321826934814453125e-10‬f
#define float_inv_nmax_ulong_adj 2.7105054312137610850186320021749e-20‬f

/*
Mantissa of random doubles can be generated from one ulong or a pair of uint
From ulong, just and the ulong with 0x000fffffffffffff to extract the lower 52-bits
A pair of uint can be thought of as a ulong. To extract the lower 52-bits, we keep
the uint representing the lower 32-bits, and extract the lower 20-bits from the
other uint. So we need, in total, 2 masks. One to extract lower 23-bits for
generating floats, and another to extract lower 20-bits for generating doubles
*/
// 0x000fffff
#define LOWER_20b_MASK_uint 1048575
// 0x007fffff
#define LOWER_23b_MASK_uint ‭8388607‬
// 0xffffffff
#define UINT_FFFFFFFF 4294967295‬;
// 0x3fffffff
#define UINT_3FFFFFFF ‭1073741823‬;
// 

static inline cl_float simple_cc_01_uint(cl_uint x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_max_uint;
} // simple_cc_01_uint

static inline cl_float simple_cc_01_ulong(cl_ulong x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_max_ulong;
} // simple_cc_01_ulong

static inline cl_float simple_co_01_uint(cl_uint x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_nmax_uint;
} // simple_co_01_uint

static inline cl_float simple_co_01_ulong(cl_ulong x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_nmax_ulong;
} // simple_co_01_ulong

static inline cl_float simple_oc_01_uint(cl_uint x) {
    cl_float tmp = (cl_float)(x);
	return (tmp+1.0f)*float_inv_nmax_uint;
} // simple_oc_01_uint

static inline cl_float simple_oc_01_ulong(cl_ulong x) {
    cl_float tmp = (cl_float)(x);
	return (tmp+1.0f)*float_inv_nmax_ulong;
} // simple_oc_01_ulong

static inline cl_float simple_oo_01_uint(cl_uint x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_nmax_uint + float_inv_nmax_uint_adj;
} // simple_oo_01_uint

static inline cl_float simple_oo_01_ulong(cl_ulong x) {
    cl_float tmp = (cl_float)(x);
	return tmp*float_inv_nmax_ulong + float_inv_nmax_ulong_adj;
} // simple_oo_01_ulong

/*
The functions to calculate the the exponent for single- and double-precision random
numbers in the interval [0, 1) are defined as flt_exponent_co_01() and
dbl_exponent_co_01()

Random numbers in the interval (0, 1] can be obtained from random numbers generated
in the interval [0, 1) by converting any 0.0 to 1.0

Random numbers in the interval (0, 1) can be obtained from random numbers generated
in the interval [0, 1) by discarding all 0.0

Random numbers in the interval [0, 1] can be obtained from random numbers generated
in the interval [0, 1) by randomly converting 0.0 to 1.0. At least two bits in the
random bit streams given to the functions flt_exponent_co_01() and
dbl_exponent_co_01() remain unused. One of them can be used to decide whether the
0.0 gets flipped to 1.0 or remains as 0.0
*/
static inline cl_uint flt_exponent_co_01(cl_uint[4] x) {
	// Initialize the exponents to be in the correct bit positions
	cl_uint outexp=126; // Start exponent at +126 and count down
	cl_uint delta=1;
	cl_uint step=32;
	cl_uint idx, tmpbits;
    cl_bool chk=true;

	// Iterate through the array of uint....
	//     outexp will not reach 0 in this for loop
	for (idx = 0; idx < 3; idx++) {
		if (x[idx] == ‭UINT_FFFFFFFF‬) {  // If 32-bit pattern is all ones...
			outexp -= step;
		} else {
			tmpbits = x[idx];
			do {
				if (tmpbits & 0x00000001) {
					tmpbits >>= 1;
					outexp -= delta;
				} else {
		            // Early termination
					chk = false;
					break;
				}
			} while (outexp > 0);
		}
		// Early termination
		if (chk == false) {
			break;
		}
	}
	if (chk) { // Only enter this if condition if we need to check the last uint
		tmpbits = x[3];
		if (tmpbits >= UINT_3FFFFFFF) {
			outexp = 0;
		} else {
			do {
				if (tmpbits & 0x00000001) {
					tmpbits >>= 1;
					outexp -= delta;
				} else {
					chk = false;
					break;
				}
			} while (outexp > 0);
		}
	}
	return outexp << 23;
} // flt_exponent_co_01

static inline cl_uint dbl_exponent_co_01(cl_uint[32] x) {
	// Initialize the exponents to be in the correct bit positions
	cl_uint outexp=1022;
	cl_uint delta=1; // Start exponent at +1022 and count down
	cl_uint step=32;
	cl_uint idx, tmpbits;
    cl_bool chk=true;

	// Iterate through the array of uint....
	//     outexp will not reach 0 in this for loop
	for (idx = 0; idx < 31; idx++) {
		if (x[idx] == UINT_FFFFFFFF) {  // If 32-bit pattern is all ones...
			outexp -= step;
		} else {
			tmpbits = x[idx];
			do {
				if (tmpbits & 0x00000001) {
					tmpbits >>= 1;
					outexp -= delta;
				} else {
		            // Early termination... break out of while loop
					chk = false;
					break;
				}
			} while (outexp > 0);
		}
		// Early termination (break out of for loop)
		if (chk == false) {
			break;
		}
	}
	if (chk) { // Only enter this if condition if we need to check the last uint
		tmpbits = x[32];
		if (tmpbits >= UINT_3FFFFFFF) {
			outexp = 0;
		} else {
			do {
				if (tmpbits & 0x00000001) {
					tmpbits >>= 1;
					outexp -= delta;
				} else {
					chk = false;
					break;
				}
			} while (chk > 0);
		}
	}
	return outexp << 20;
} // dbl_exponent_co_01

/*
Define function to calculate inverse CDF of standard Gaussian distribution.
Taken from PhD thesis of Thomas Luu (Department of Mathematics at Universty College
of London). These are the same as the hybrid approximation functions in the thesis.
*/
static inline cl_float normcdfinv_float(cl_float u) {
	cl_float	v, p, q, ushift, tmp;

	tmp = u;

	if (u < 0.0f || u > 1.0f) {
		return NAN;
	}
	if (u <= 0.0f) {
		return FLT_MIN;// Float.NEGATIVE_INFINITY;
	}
	if (u >= 1.0f) {
		return FLT_MAX; // Float.POSITIVE_INFINITY;
	}

	ushift = tmp - 0.5f;

	v = copysign(ushift, 0.0f);
	
	if (v < 0.499433f) {
		v = rsqrt((-tmp*tmp) + tmp);
		v *= 0.5f;

		p = 0.001732781974270904f;
		p = p * v + 0.1788417306083325f;
		p = p * v + 2.804338363421083f;
		p = p * v + 9.35716893191325f;
		p = p * v + 5.283080058166861f;
		p = p * v + 0.07885390444279965f;
		p *= ushift;

		q = 0.0001796248328874524f;
		q = q * v + 0.02398533988976253f;
		q = q * v + 0.4893072798067982f;
		q = q * v + 2.406460595830034f;
		q = q * v + 3.142947488363618f;
	} else {
		if (ushift > 0.0f) {
			tmp = 1.0f - tmp;
		}
		v = log2(tmp+tmp);
		v *= -0.6931471805599453f;
		if (v < 22.0f) {
			p = 0.000382438382914666f;
			p = p * v + 0.03679041341785685f;
			p = p * v + 0.5242351532484291f;
			p = p * v + 1.21642047402659f;

			q = 9.14019972725528e-6f;
			q = q * v + 0.003523083799369908f;
			q = q * v + 0.126802543865968f;
			q = q * v + 0.8502031783957995f;
		} else {
			p = 0.00001016962895771568f;
			p = p * v + 0.003330096951634844f;
			p = p * v + 0.1540146885433827f;
			p = p * v + 1.045480394868638f;

			q = 1.303450553973082e-7f;
			q = q * v + 0.0001728926914526662f;
			q = q * v + 0.02031866871146244f;
			q = q * v + 0.3977137974626933f;
		}
		p *= copysign(v, ushift);
	}
	q = q * v + 1.0f;
	v = 1.0f / q;
	return p * v;
} // normcdfinv_float

)EOK";