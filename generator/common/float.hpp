const char * flt_kernel_util_defs = R"EOK(
////////////////////////////////////////////////////////////////////////////////////
#define float_inv_max_uint  2.328306437080797e-10f
#define float_inv_nmax_uint 0x2f800000

#define float_inv_max_ulong 5.4210108624275221700372640e-20f
#define float_inv_nmax_ulong 0x1f800000

#define float_inv_nmax_uint_adj 0x2f000000
#define float_inv_nmax_ulong_adj 0x1f000000

inline float simple_cc_01_uint(uint x) {
	float tmp = (float)(x);
	return  tmp*float_inv_max_uint;
} // simple_cc_01_uint

inline float simple_cc_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*float_inv_max_ulong;
} // simple_cc_01_ulong

inline float simple_co_01_uint(uint x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_uint);
} // simple_co_01_uint

inline float simple_co_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_ulong);
} // simple_co_01_ulong

inline float simple_oc_01_uint(uint x) {
    float tmp = (float)(x);
	return (tmp+1.0f)*as_float(float_inv_nmax_uint);
} // simple_oc_01_uint

inline float simple_oc_01_ulong(ulong x) {
    float tmp = (float)(x);
	return (tmp+1.0f)*as_float(float_inv_nmax_ulong);
} // simple_oc_01_ulong

inline float simple_oo_01_uint(uint x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_uint) + as_float(float_inv_nmax_uint_adj);
} // simple_oo_01_uint

inline float simple_oo_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_ulong) + as_float(float_inv_nmax_ulong_adj);
} // simple_oo_01_ulong
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
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

Mantissa of random doubles can be generated from one ulong or a pair of uint
From ulong, just and the ulong with 0x000fffffffffffff to extract the lower 52-bits
A pair of uint can be thought of as a ulong. To extract the lower 52-bits, we keep
the uint representing the lower 32-bits, and extract the lower 20-bits from the
other uint. So we need, in total, 2 masks. One to extract lower 23-bits for
generating floats, and another to extract lower 20-bits for generating doubles
*/
////////////////////////////////////////////////////////////////////////////////////

// 0x000fffff
#define LOWER_20b_MASK_uint 1048575
// 0x007fffff
#define LOWER_23b_MASK_uint ‭8388607‬
// 0xffffffff
#define UINT_FFFFFFFF 4294967295‬
// 0x3fffffff
#define UINT_3FFFFFFF ‭1073741823‬

// Kernel should iterate through uint that are equal to 0xffffffff to adjust its
// copy of the exponent. Once the kernel finds an uint that is not 0xffffffff, it
// then calls this function to determine the correct exponent, and bit shift
// accordingly
inline uint exponent_adj(uint inBits, uint inexp) {
	uint outexp=inexp; // Start exponent at the input and count down
	uint idx, tmpbits;

    if (outexp > 0) {
		for (idx = 32; idx > 0; idx--) {
			if ((outexp == 0) || ((tmpbits & 0x00000001) == 0)) {
				break;
			}
			outexp--;
			tmpbits >>= 1;
		}
	}

    return outexp;
} // exponent_adj

////////////////////////////////////////////////////////////////////////////////////
/*
Define function to calculate inverse CDF of standard Gaussian distribution.
Taken from PhD thesis of Thomas Luu (Department of Mathematics at Universty College
of London). These are the same as the hybrid approximation functions in the thesis.
*/
////////////////////////////////////////////////////////////////////////////////////
inline float normcdfinv_float(float u) {
	float	v, p, q, ushift, tmp;

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
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
/*
Kernel functions for fast conversion of uint to float while copying between buffers
*/
////////////////////////////////////////////////////////////////////////////////////
kernel void CopyUintAsFlt01CC(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_cc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01CO(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_co_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01OC(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01OO(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oo_01_uint(src[ii]);
	}
}

// Kernel functions for fast conversion of ulong to float while copying between buffers
kernel void CopyUlongAsFlt01CC(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_cc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01CO(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_co_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01OC(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01OO(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oo_01_ulong(src[ii]);
	}
}

// Kernel functions for fast copy of Gaussian float while copying between buffers
kernel void CopyUintAsNormFlt01OO(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	float localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = simple_oo_01_uint(src[ii]);
		dst[ii] = normcdfinv_float(localVal);
	}
}

kernel void CopyUlongAsNormFlt01OO(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	float localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = simple_oo_01_ulong(src[ii]);
		dst[ii] = normcdfinv_float(localVal);
	}
}
////////////////////////////////////////////////////////////////////////////////////

)EOK";