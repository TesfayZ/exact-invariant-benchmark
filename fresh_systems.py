"""Prospective polynomial systems; disjoint from the archived ten-system study.

Two constructions: nonlinear coordinate shears of rotations (4D) and Nambu
fields from seeded polynomial pairs (3D). References are hidden from prompts.
These are planted controls, not evidence against training-data memorisation.
"""
import random
import sympy as sp
from systems import System


def shear(seed):
    rng=random.Random(seed)
    a,b,c=[rng.randint(1,4) for _ in range(3)]
    x,y,z,w=sp.symbols('x y z w', real=True)
    u=z-a*x*y
    v=w-b*x*x-c*y
    speed=1+u
    f=[-y*speed,x*speed,a*(x*x-y*y)*speed,(-2*b*x*y+c*x)*speed]
    return System(f'planted_shear_{seed}','planted',[x,y,z,w],[],
                  field=list(map(sp.expand,f)),conserved_ref=[u,v,x*x+y*y],
                  description='A polynomial autonomous system in the declared coordinates.')


def nambu(seed):
    rng=random.Random(seed)
    x,y,z=sp.symbols('x y z',real=True)
    a,b,c,d=[rng.randint(1,5) for _ in range(4)]
    p=x*x+a*y*y+b*z*z+x*y
    q=z+c*x*y+d*y*y*z
    grad=lambda f: sp.Matrix([sp.diff(f,v) for v in (x,y,z)])
    f=grad(p).cross(grad(q))
    return System(f'planted_nambu_{seed}','planted',[x,y,z],[],
                  field=list(map(sp.expand,f)),conserved_ref=[p,q],
                  description='A polynomial autonomous system in the declared coordinates.')


REGISTRY={**{f'planted_shear_{s}':(lambda s=s:shear(s)) for s in (11,23,47)},
          **{f'planted_nambu_{s}':(lambda s=s:nambu(s)) for s in (11,23,47)}}
