"""CEC 2020 multimodal multiobjective (MMO) benchmark problems.

The 24 entries follow the supplied MATLAB ``main.m`` experiment list. Inputs
may be a decision vector or a matrix of decision vectors; objectives are
returned in the corresponding shape.
"""

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class MMOProblem:
    number: int
    name: str
    lower_bounds: tuple
    upper_bounds: tuple
    objectives: int

    @property
    def variables(self):
        return len(self.lower_bounds)


_SPECS = (
    ("MMF1", (1, -1), (3, 1), 2), ("MMF2", (0, 0), (1, 2), 2),
    ("MMF4", (-1, 0), (1, 2), 2), ("MMF5", (1, -1), (3, 3), 2),
    ("MMF7", (1, -1), (3, 1), 2), ("MMF8", (-pi, 0), (pi, 9), 2),
    ("MMF10", (.1, .1), (1.1, 1.1), 2), ("MMF11", (.1, .1), (1.1, 1.1), 2),
    ("MMF12", (0, 0), (1, 1), 2), ("MMF13", (.1, .1, .1), (1.1, 1.1, 1.1), 2),
    ("MMF14", (0, 0, 0), (1, 1, 1), 3), ("MMF15", (0, 0, 0), (1, 1, 1), 3),
    ("MMF1_e", (1, -20), (3, 20), 2), ("MMF14_a", (0, 0, 0), (1, 1, 1), 3),
    ("MMF15_a", (0, 0, 0), (1, 1, 1), 3), ("MMF10_l", (.1, .1), (1.1, 1.1), 2),
    ("MMF11_l", (.1, .1), (1.1, 1.1), 2), ("MMF12_l", (0, 0), (1, 1), 2),
    ("MMF13_l", (.1, .1, .1), (1.1, 1.1, 1.1), 2), ("MMF15_l", (0, 0, 0), (1, 1, 1), 3),
    ("MMF15_a_l", (0, 0, 0), (1, 1, 1), 3), ("MMF16_l1", (0, 0, 0), (1, 1, 1), 3),
    ("MMF16_l2", (0, 0, 0), (1, 1, 1), 3), ("MMF16_l3", (0, 0, 0), (1, 1, 1), 3),
)
CEC2020_MMO_PROBLEMS = tuple(MMOProblem(i, *spec) for i, spec in enumerate(_SPECS, 1))
MMO_PROBLEMS = {p.name: p for p in CEC2020_MMO_PROBLEMS}


def _sphere_objectives(x, m, g):
    angles = x[:, :m - 1] * (pi / 2)
    cos_products = np.fliplr(np.cumprod(np.column_stack((np.ones(len(x)), np.cos(angles))), axis=1))
    sin_values = np.column_stack((np.ones(len(x)), np.sin(x[:, m - 2::-1] * (pi / 2))))
    return (1 + g[:, None]) * cos_products * sin_values


def _g_peak(x, n=2):
    return 2 - np.exp(-2 * np.log10(2) * ((x - .1) / .8) ** 2) * np.sin(n * pi * x) ** 2


def evaluate_mmo(x, problem, objectives=None, num_of_peak=2):
    """Evaluate a CEC2020 MMO problem by its name or 1-based suite number."""
    if isinstance(problem, (int, np.integer)) and not 1 <= int(problem) <= len(CEC2020_MMO_PROBLEMS):
        raise ValueError("problem must be a suite number 1..24 or a problem name")
    spec = CEC2020_MMO_PROBLEMS[int(problem) - 1] if isinstance(problem, (int, np.integer)) else MMO_PROBLEMS.get(str(problem))
    if spec is None:
        raise ValueError("problem must be a suite number 1..24 or a problem name")
    a = np.asarray(x, dtype=float)
    single = a.ndim == 1
    if single:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != spec.variables or not np.all(np.isfinite(a)):
        raise ValueError(f"{spec.name} expects finite decision vectors with {spec.variables} variables")
    if np.any(a < spec.lower_bounds) or np.any(a > spec.upper_bounds):
        raise ValueError(f"decision vectors for {spec.name} are outside the published bounds")
    x1, x2 = a[:, 0], a[:, 1] if a.shape[1] > 1 else (a[:, 0], None)
    n = len(a)
    if spec.name == "MMF1":
        q = np.abs(x1 - 2); y = np.c_[q, 1 - np.sqrt(q) + 2 * (x2 - np.sin(6*pi*q+pi))**2]
    elif spec.name == "MMF1_e":
        q = np.abs(x1 - 2); target = np.sin(6*pi*q+pi); target = np.where(x1 >= 2, np.exp(x1)*target, target)
        y = np.c_[q, 1-np.sqrt(q)+2*(x2-target)**2]
    elif spec.name == "MMF2":
        z = np.where(x2 > 1, x2-1, x2)-np.sqrt(x1); y=np.c_[x1,1-np.sqrt(x1)+2*(4*z*z-2*np.cos(20*z*pi/np.sqrt(2))+2)]
    elif spec.name == "MMF4":
        z=np.where(x2>1,x2-1,x2); y=np.c_[np.abs(x1),1-x1*x1+2*(z-np.sin(pi*np.abs(x1)))**2]
    elif spec.name == "MMF5":
        z=np.abs(x1-2); z2=np.where(x2>1,x2-2,x2); y=np.c_[z,1-np.sqrt(z)+2*(z2-np.sin(6*pi*z+pi))**2]
    elif spec.name == "MMF7":
        z=np.abs(x1-2); t=(.3*z*z*np.cos(24*pi*z+4*pi)+.6*z)*np.sin(6*pi*z+pi); y=np.c_[z,1-np.sqrt(z)+(x2-t)**2]
    elif spec.name == "MMF8":
        z=np.abs(x1); z2=np.where(x2>4,x2-4,x2); f=np.sin(z); y=np.c_[f,np.sqrt(1-f*f)+2*(z2-(f+z))**2]
    elif spec.name in ("MMF10", "MMF10_l"):
        g=2-np.exp(-((x2-.2)/.004)**2)-.8*np.exp(-((x2-.6)/.4)**2); y=np.c_[x1,g/x1]
    elif spec.name in ("MMF11", "MMF11_l"):
        y=np.c_[x1,_g_peak(x2,2)/x1]
    elif spec.name in ("MMF12", "MMF12_l"):
        g=_g_peak(x2,2); f1=x1; y=np.c_[f1,g*(1-(f1/g)**2-(f1/g)*np.sin(8*pi*f1))]
    elif spec.name in ("MMF13", "MMF13_l"):
        g=2-np.exp(-2*np.log10(2)*((x2+np.sqrt(a[:,2])-.1)/.8)**2)*np.sin(2*pi*(x2+np.sqrt(a[:,2])))**6; y=np.c_[x1,g/x1]
    else:
        m = spec.objectives if objectives is None else int(objectives)
        if m < 2 or a.shape[1] < m: raise ValueError("objectives must be at least 2 and no greater than the variable count")
        t=a[:,-1]
        if spec.name == "MMF14": g=2-np.sin(num_of_peak*pi*t)**2
        elif spec.name == "MMF14_a": g=2-np.sin(num_of_peak*pi*(a[:,-1]-.5*np.sin(pi*a[:,-2])+1/(2*num_of_peak)))**2
        elif spec.name in ("MMF15","MMF15_a","MMF15_l","MMF15_a_l"):
            z=t if spec.name.endswith("_l") and spec.name!="MMF15_a_l" else t-.5*np.sin(pi*a[:,-2])
            if spec.name in ("MMF15_a","MMF15_a_l"): z=z+1/(2*num_of_peak)
            g=_g_peak(z,num_of_peak)
        else:
            global_peaks = 1 if spec.name.endswith("l2") else 2
            local_peaks = 1 if spec.name.endswith("l1") else 2
            g=np.where(t<.5,2-np.sin(2*global_peaks*pi*t)**2,_g_peak(t,local_peaks))
        y=_sphere_objectives(a,m,g)
    return y[0] if single else y


def make_mmo(problem, **options):
    """Return an objective callable for a CEC2020 MMO problem."""
    return lambda x: evaluate_mmo(x, problem, **options)
