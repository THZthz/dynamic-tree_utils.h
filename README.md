# dynamic-tree_utils.h
Rewrites "dtVec" of the header file "utils.h" from erincatto's repo "dynamic-tree".

Original issue: https://github.com/erincatto/dynamic-tree/issues/2.

There is already a builtin operator+ for vector types with the generic vector support in GCC.

So overload operators of __m128 will report errors.

Wrap `__m128` into a class will solve the problem.
