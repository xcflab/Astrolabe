import ast
from functools import partial

import numpy as np


_POSE_SELECTOR_VARS = {
    "sse",
    "cdr",
    "bfactor",
    "interface",
    "surface",
    "surface_high_sap",
    "buried_unsat",
    "cavity_neighbor",
    "ligand_unsat",
    "ligand_contact",
    "membrane_outer_surface",
    "long_helix_core",
    "long_loop",
}


class _SelectorExpressionTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            op_cls = ast.BitAnd
        elif isinstance(node.op, ast.Or):
            op_cls = ast.BitOr
        else:
            return node

        expr = values[0]
        for value in values[1:]:
            expr = ast.copy_location(
                ast.BinOp(left=expr, op=op_cls(), right=value), node
            )
        return expr

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(op=ast.Invert(), operand=operand), node)
        return ast.copy_location(ast.UnaryOp(op=node.op, operand=operand), node)


def construct_position_selector(
    expression, evaluation_result, protomers
):
    default_sites = {name: np.ones(protomer.length, dtype=bool) for name, protomer in protomers.items()}
    if expression == "None" or expression is None:
        return default_sites

    position_vars = extract_variables(expression)
    complex_poses = getattr(evaluation_result, "complex_poses", None)
    if complex_poses is None:
        if position_vars & _POSE_SELECTOR_VARS:
            complex_paths = getattr(evaluation_result, "complex_paths", None)
            if not complex_paths:
                raise ValueError(
                    "Position selector expression requires evaluated complex paths "
                    "to construct ComplexPoses."
                )
            from astrolabe.evaluators.utils import ComplexPoses

            path_only_vars = {"membrane_outer_surface"}
            if position_vars <= path_only_vars:
                complex_poses = ComplexPoses({})
                complex_poses.complex_paths = dict(complex_paths)
            else:
                complex_poses = ComplexPoses.construct_complex_poses(evaluation_result)
                complex_poses.complex_paths = dict(complex_paths)
            try:
                evaluation_result.complex_poses = complex_poses
            except Exception:
                pass
        else:
            from astrolabe.evaluators.utils import ComplexPoses

            complex_poses = ComplexPoses({})
    else:
        complex_paths = getattr(evaluation_result, "complex_paths", None)
        if complex_paths and not hasattr(complex_poses, "complex_paths"):
            complex_poses.complex_paths = dict(complex_paths)

    env = {}
    for var in position_vars:
        if var == "sse":
            from astrolabe.selectors.sse import SecondaryStructurePositionSelector

            env[var] = partial(
                SecondaryStructurePositionSelector, complex_poses, merge="any"
            )
        elif var == "cdr":
            from astrolabe.selectors.cdr import CDRPositionSelector

            env[var] = partial(CDRPositionSelector, complex_poses, merge="any")
        elif var == "bfactor":
            from astrolabe.selectors.bfactor import BFactorPositionSelector

            env[var] = partial(BFactorPositionSelector, complex_poses, merge="any")
        elif var == "interface":
            from astrolabe.selectors.interface import InterfacePositionSelector

            env[var] = partial(InterfacePositionSelector, complex_poses, merge="any")
        elif var == "surface":
            from astrolabe.selectors.surface import SurfacePositionSelector

            env[var] = partial(SurfacePositionSelector, complex_poses, merge="any")
        elif var == "surface_high_sap":
            from astrolabe.selectors.protein_quality import (
                SurfaceHighSAPPositionSelector,
            )

            env[var] = partial(
                SurfaceHighSAPPositionSelector, complex_poses, merge="any"
            )
        elif var == "buried_unsat":
            from astrolabe.selectors.protein_quality import (
                BuriedUnsatPositionSelector,
            )

            env[var] = partial(BuriedUnsatPositionSelector, complex_poses, merge="any")
        elif var == "cavity_neighbor":
            from astrolabe.selectors.protein_quality import (
                CavityNeighborPositionSelector,
            )

            env[var] = partial(
                CavityNeighborPositionSelector, complex_poses, merge="any"
            )
        elif var == "ligand_unsat":
            from astrolabe.selectors.ligand import (
                LigandContactPositionSelector,
                LigandUnsatisfiedPolarPositionSelector,
            )

            env[var] = partial(
                LigandUnsatisfiedPolarPositionSelector, complex_poses, merge="any"
            )
            env["ligand_contact"] = partial(
                LigandContactPositionSelector, complex_poses, merge="any"
            )
        elif var == "ligand_contact":
            from astrolabe.selectors.ligand import LigandContactPositionSelector

            env[var] = partial(
                LigandContactPositionSelector, complex_poses, merge="any"
            )
        elif var == "membrane_outer_surface":
            from astrolabe.selectors.ligand import MembraneOuterSurfacePositionSelector

            env[var] = partial(
                MembraneOuterSurfacePositionSelector, complex_poses, merge="any"
            )
        elif var == "hotspot":
            from astrolabe.selectors.hotspot import HotspotAliasPositionSelector

            env[var] = partial(HotspotAliasPositionSelector, complex_poses, merge="any")
        elif var == "hole_surface":
            from astrolabe.selectors.hole import HoleSurfacePositionSelector

            env[var] = partial(
                HoleSurfacePositionSelector, complex_poses, merge="any"
            )
        elif var == "pore_bottleneck":
            from astrolabe.selectors.hole import PoreBottleneckPositionSelector

            env[var] = partial(
                PoreBottleneckPositionSelector, complex_poses, merge="any"
            )
        elif var == "long_helix_core":
            from astrolabe.selectors.sse import LongHelixCorePositionSelector

            env[var] = partial(LongHelixCorePositionSelector, complex_poses, merge="any")
        elif var == "long_loop":
            from astrolabe.selectors.sse import (
                LongLoopPositionSelector,
            )

            env[var] = partial(
                LongLoopPositionSelector, complex_poses, merge="any"
            )
        elif var == "rosetta_interface_energy":
            from astrolabe.selectors.rosetta_interface_energy import (
                RosettaInterfaceEnergyPositionSelector,
            )

            env[var] = partial(
                RosettaInterfaceEnergyPositionSelector,
                evaluation_result,
                protomers,
            )
        else:
            raise ValueError(f"Invalid variable: {var}")

    tree = ast.parse(expression, mode="eval")
    tree = _SelectorExpressionTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    eval_globals = {"__builtins__": {}}
    eval_globals.update(env)
    final_selector = eval(compile(tree, "<position_selector>", "eval"), eval_globals)


    if final_selector is None:
        return default_sites

    if isinstance(final_selector, bool):
        if final_selector:
            return default_sites
        return {
            proto: np.zeros_like(mask, dtype=bool)
            for proto, mask in default_sites.items()
        }

    result = {}
    for proto, base_mask in default_sites.items():
        selector_mask = final_selector.sites.get(proto)
        if selector_mask is None:
            result[proto] = base_mask.copy()
        else:
            result[proto] = np.logical_and(base_mask, selector_mask)
    return result


def extract_variables(expr):
    return {node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)}
