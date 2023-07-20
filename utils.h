#pragma once

#include <math.h>
#include <memory.h>
#include <immintrin.h>

static const float dtPi = 3.141592654f;

inline int dtMin(int a, int b)
{
  return a < b ? a : b;
}

inline int dtMax(int a, int b)
{
  return a > b ? a : b;
}

inline int dtClamp(int a, int b, int c)
{
  return a < b ? b : (c < a ? c : a);
}

inline int dtAbs(int a)
{
  return a > 0 ? a : -a;
}

inline float dtAbs(float a)
{
  return a > 0.0f ? a : -a;
}

template<typename T>
inline void dtSwap(T& a, T& b)
{
  T temp = a;
  a = b;
  b = temp;
}

class dtVec
{
public:

  inline dtVec() = default;
  inline dtVec(float f0, float f1, float f2, float f3) : m_value(_mm_setr_ps(f0,f1,f2,f3)) {}

  // Implicit conversions (so float can be converted to dtVec):
  inline dtVec(float f) : m_value(_mm_set1_ps(f)) {}
  inline dtVec(const __m128& rhs) : m_value(rhs) {}

  inline dtVec& operator=(const __m128& rhs)
  {
    m_value = rhs;
    return *this;
  }

  inline explicit operator __m128() const { return m_value; }

  // We don't need this.
  // vector4f(const vector4f&) {}
  // operator=(const vector4f&) {}

  __m128 m_value{};
};

static const dtVec dtVec_Zero = { 0.0f, 0.0f, 0.0f, 0.0f };
static const dtVec dtVec_UnitX = { 1.0f, 0.0f, 0.0f, 0.0f };
static const dtVec dtVec_UnitY = { 0.0f, 1.0f, 0.0f, 0.0f };
static const dtVec dtVec_UnitZ = { 0.0f, 0.0f, 1.0f, 0.0f };

inline dtVec dtVecSet(float x, float y, float z)
{
  return _mm_set_ps(0.0f, z, y, x);
}

inline dtVec dtVecSet(float x, float y, float z, float w)
{
  return _mm_set_ps(w, z, y, x);
}

inline dtVec dtVecSet(dtVec& x, dtVec& y, dtVec& z)
{
  dtVec r;
  r = _mm_unpacklo_ps(x.m_value, y.m_value);	// (x, y, x, y)
  r = _mm_movelh_ps(r.m_value, z.m_value);	// (x, y, z, z)
  return r;
}

inline dtVec dtSplat(float x)
{
  return _mm_set1_ps(x);
}

inline dtVec dtSetX(const dtVec& v, float x)
{
  dtVec t = _mm_set_ss(x);
  return _mm_move_ss(v.m_value, t.m_value);
}

inline dtVec dtSetY(const dtVec& v, float y)
{
  dtVec r = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 0, 1));
  dtVec t = _mm_set_ss(y);
  r = _mm_move_ss(r.m_value, t.m_value);
  r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 2, 0, 1));
  return r;
}

inline dtVec dtSetZ(const dtVec& v, float z)
{
  dtVec r = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 0, 1, 2));
  dtVec t = _mm_set_ss(z);
  r = _mm_move_ss(r.m_value, t.m_value);
  r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 0, 1, 2));
  return r;
}

inline dtVec dtSetW(const dtVec& v, float w)
{
  dtVec r = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 2, 1, 3));
  dtVec t = _mm_set_ss(w);
  r = _mm_move_ss(r.m_value, t.m_value);
  r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 2, 1, 3));
  return r;
}

inline float dtGetX(const dtVec& v)
{
  float s;
  _mm_store_ss(&s, v.m_value);
  return s;
}

inline float dtGetY(const dtVec& v)
{
  dtVec t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
  float s;
  _mm_store_ss(&s, t.m_value);
  return s;
}

inline float dtGetZ(const dtVec& v)
{
  dtVec t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
  float s;
  _mm_store_ss(&s, t.m_value);
  return s;
}

inline float dtGetW(const dtVec& v)
{
  dtVec t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
  float s;
  _mm_store_ss(&s, t.m_value);
  return s;
}

inline float dtGet(const dtVec& v, int index)
{
  switch (index)
  {
  case 0:
    return dtGetX(v);
  case 1:
    return dtGetY(v);
  case 2:
    return dtGetZ(v);
  default:
    return dtGetW(v);
  }
}

inline dtVec dtMin(const dtVec& a, const dtVec& b)
{
  return _mm_min_ps(a.m_value, b.m_value);
}

inline dtVec dtMax(const dtVec& a, const dtVec& b)
{
  return _mm_max_ps(a.m_value, b.m_value);
}

inline dtVec dtAbs(const dtVec& a)
{
  return _mm_max_ps(a.m_value, -a.m_value);
}

inline dtVec operator + (const dtVec& a, const dtVec& b)
{
  return _mm_add_ps(a.m_value, b.m_value);
}

inline dtVec operator - (const dtVec& a, const dtVec& b)
{
  return _mm_sub_ps(a.m_value, b.m_value);
}

inline dtVec& operator += (dtVec& a, const dtVec& b)
{
  a = _mm_add_ps(a.m_value, b.m_value);
  return a;
}

inline dtVec& operator -= (dtVec& a, const dtVec& b)
{
  a = _mm_sub_ps(a.m_value, b.m_value);
  return a;
}

inline dtVec operator * (float a, const dtVec& b)
{
  dtVec av = _mm_set1_ps(a);
  return _mm_mul_ps(av.m_value, b.m_value);
}

inline dtVec operator * (dtVec& a, const dtVec& b)
{
  return _mm_mul_ps(a.m_value, b.m_value);
}

// TODO
inline dtVec operator * (dtVec& a, dtVec& b)
{
  return _mm_mul_ps(a.m_value, b.m_value);
}

// TODO
inline dtVec operator * (const dtVec& a,const dtVec& b)
{
  return _mm_mul_ps(a.m_value, b.m_value);
}

inline dtVec operator - (const dtVec& a)
{
  return _mm_sub_ps(_mm_setzero_ps(), a.m_value);
}

inline bool operator == (const dtVec& a, const dtVec& b)
{
  dtVec t = _mm_cmpeq_ps(a.m_value, b.m_value);
  return _mm_movemask_ps(t.m_value) == 0xF;
}

inline dtVec dtDot3(const dtVec& a, const dtVec& b)
{
  dtVec t = _mm_mul_ps(a.m_value, b.m_value);
  dtVec xx = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 0));
  dtVec yy = _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1));
  dtVec zz = _mm_shuffle_ps(t, t, _MM_SHUFFLE(2, 2, 2, 2));
  return _mm_add_ps(_mm_add_ps(xx.m_value, yy.m_value), zz.m_value);
}

inline dtVec dtCross(const dtVec& a, const dtVec& b)
{
  // http://threadlocalmutex.com/?p=8
  dtVec a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
  dtVec b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
  dtVec c = _mm_sub_ps(_mm_mul_ps(a.m_value, b_yzx.m_value), _mm_mul_ps(a_yzx.m_value, b.m_value));
  return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

inline dtVec dtLength3(const dtVec& a)
{
  dtVec t = dtDot3(a, a);
  return _mm_sqrt_ps(t.m_value);
}

inline dtVec dtNormalize3(const dtVec& v)
{
  dtVec length = _mm_sqrt_ps(dtDot3(v, v).m_value);
  return _mm_div_ps(v.m_value, length.m_value);
}

struct dtMtx
{
  dtVec cx, cy, cz, cw;
};

inline dtVec dtTransformVector(const dtMtx& m, const dtVec& v)
{
  dtVec x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
  dtVec y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
  dtVec z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

  dtVec r = _mm_mul_ps(m.cx.m_value, x.m_value);
  r = _mm_add_ps(_mm_mul_ps(m.cy.m_value, y.m_value), r.m_value);
  r = _mm_add_ps(_mm_mul_ps(m.cz.m_value, z.m_value), r.m_value);
  return r;
}

inline dtVec dtTransformPoint(const dtMtx& m, const dtVec& p)
{
  dtVec x = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 0, 0, 0));
  dtVec y = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1, 1, 1, 1));
  dtVec z = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 2, 2));

  dtVec r = _mm_mul_ps(m.cx.m_value, x.m_value);
  r = _mm_add_ps(_mm_mul_ps(m.cy.m_value, y.m_value), r.m_value);
  r = _mm_add_ps(_mm_mul_ps(m.cz.m_value, z.m_value), r.m_value);
  r = _mm_add_ps(m.cw.m_value, r.m_value);
  return r;
}

inline dtVec dtInvTransformVector(const dtMtx& m, const dtVec& v)
{
  dtVec x = dtDot3(m.cx, v);
  dtVec y = dtDot3(m.cy, v);
  dtVec z = dtDot3(m.cz, v);
  return dtVecSet(x, y, z);
}

inline dtVec dtInvTransformPoint(const dtMtx& m, const dtVec& p)
{
  dtVec v = _mm_sub_ps(p.m_value, m.cw.m_value);
  dtVec x = dtDot3(m.cx, v);
  dtVec y = dtDot3(m.cy, v);
  dtVec z = dtDot3(m.cz, v);
  return dtVecSet(x, y, z);
}

inline dtMtx dmTranspose33(const dtMtx& a)
{
  dtVec t1 = _mm_shuffle_ps(a.cx, a.cy, _MM_SHUFFLE(0, 0, 0, 0));
  dtVec t2 = _mm_shuffle_ps(a.cx, a.cy, _MM_SHUFFLE(1, 1, 1, 1));
  dtVec t3 = _mm_shuffle_ps(a.cx, a.cy, _MM_SHUFFLE(2, 2, 2, 2));

  dtMtx b;
  b.cx = _mm_shuffle_ps(t1, a.cz, _MM_SHUFFLE(3, 0, 2, 0));
  b.cy = _mm_shuffle_ps(t2, a.cz, _MM_SHUFFLE(3, 1, 2, 0));
  b.cz = _mm_shuffle_ps(t3, a.cz, _MM_SHUFFLE(3, 2, 2, 0));
  b.cw = _mm_setzero_ps();
  return b;
}

// y = R * x + p
// x = RT * (y - p)
//   = RT * y - RT * p
inline dtMtx dtMtx_InvertOrtho(const dtMtx& m)
{
  dtMtx im = dmTranspose33(m);
  im.cw = -dtTransformVector(im, m.cw);
  return im;
}

struct dtAABB
{
  dtVec lowerBound;
  dtVec upperBound;
};

inline dtAABB dtUnion(const dtAABB& a, const dtAABB& b)
{
  dtAABB c;
  c.lowerBound = dtMin(a.lowerBound, b.lowerBound);
  c.upperBound = dtMax(a.upperBound, b.upperBound);
  return c;
}

inline float dtArea(const dtAABB& a)
{
  dtVec w = a.upperBound - a.lowerBound;
  dtVec x = _mm_shuffle_ps(w, w, _MM_SHUFFLE(0, 0, 0, 0));
  dtVec y = _mm_shuffle_ps(w, w, _MM_SHUFFLE(1, 1, 1, 1));
  dtVec z = _mm_shuffle_ps(w, w, _MM_SHUFFLE(2, 2, 2, 2));
  dtVec area = x * y + y * z + z * x;
  area = area + area;

  float s;
  _mm_store_ss(&s, area.m_value);
  return s;
}

inline dtVec dtCenter(const dtAABB& a)
{
  return dtSplat(0.5f).m_value * (a.lowerBound + a.upperBound).m_value;
}

inline dtVec dtExtent(const dtAABB& a)
{
  return dtSplat(0.5f).m_value * (a.upperBound - a.lowerBound).m_value;
}

struct dtFree
{
  void operator()(void* x) { free(x); }
};

class dtTimer
{
public:

  /// Constructor
  dtTimer();

  /// Reset the timer.
  void Reset();

  /// Get the time since construction or the last reset.
  float GetMilliseconds() const;

private:

#if defined(_WIN32)
  double m_start;
  static double s_invFrequency;
#endif
};
