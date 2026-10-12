"""除显式系数和实验接触模型选项外，对照历史 CLI；不加载场景或求解器。"""
import argparse
import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.scenes import flat_cli, handheld_cli
from experiments.legacy import flat_output, handheld_output
from experiments.scene import controller_config
from nmpcc.config import NormalForceLaw


def frozen_parser(variant):
    """只执行旧 main 的参数定义段，避免运行整份旧实验。"""
    from experiments import disturbance
    filename = 'exp2_nmpcc_latest.py' if variant == 'flat' else 'exp2_nmpcc_latest_grasp_pose.py'
    path = Path(__file__).resolve().parents[2] / '_archive_20261009/refactor_before' / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    definitions = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        if isinstance(node, ast.Global) or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        definitions.append(node)
    namespace = dict(argparse=argparse, K_STIFF=2793.166 if variant == 'flat' else 2419.3,
                     experiment_tools=flat_output if variant == 'flat' else handheld_output,
                     disturbance_tools=disturbance)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parser']


class CliContractTests(unittest.TestCase):
    def test_all_options_types_defaults_and_choices_match(self):
        def schema(parser, *, current=False, variant=None):
            result = []
            for action in parser._actions:
                options, choices, value_type = action.option_strings, action.choices, action.type
                if current and action.dest == 'normal_force_law':
                    self.assertEqual(action.default, 'smooth_positive')
                    self.assertEqual(action.choices, [law.value for law in NormalForceLaw])
                    continue
                if current and action.dest in ('eta_cop_y', 'If1_x'):
                    self.assertEqual(action.default, .8)
                    self.assertIs(action.type, flat_cli.unit_coefficient)
                    self.assertIsNone(action.choices)
                    if action.dest == 'If1_x':
                        self.assertEqual(set(options), {'--If1_x', '--eta_f_x'})
                        if variant == 'flat':
                            continue  # 唯一新增字段；默认值与此前 getattr(..., .8) 一致。
                        options = ['--If1_x']
                    value_type = float
                    choices = [0.8] if variant == 'handheld' else None
                result.append((action.dest, action.default, options, choices, value_type, action.nargs))
            return result
        for variant, module in [('flat', flat_cli), ('handheld', handheld_cli)]:
            with self.subTest(variant=variant):
                self.assertEqual(schema(frozen_parser(variant)),
                                 schema(module.make_parser(), current=True, variant=variant))

    def test_requested_six_static_pairs_map_to_both_arm_components(self):
        pairs = ((.8, .8), (.2, .8), (.5, .8), (.8, .2), (.8, .5), (.5, .5))
        for variant, module in [('flat', flat_cli), ('handheld', handheld_cli)]:
            parser = module.make_parser()
            scene = SimpleNamespace(variant=variant, geometry=SimpleNamespace(K_STIFF=module.K_STIFF),
                                    contact_rest_gap=.248)
            default = parser.parse_args([])
            self.assertEqual((default.eta_cop_y, default.If1_x), (.8, .8))
            for cop, force in pairs:
                for alias in ('--eta_f_x', '--If1_x'):
                    with self.subTest(variant=variant, pair=(cop, force), alias=alias):
                        args = parser.parse_args(['--eta_cop_y', str(cop), alias, str(force)])
                        self.assertEqual(args.If1_x, force)
                        self.assertFalse(hasattr(args, 'eta_f_x'))
                        config = controller_config(scene, args)
                        self.assertEqual(config.normal_force_law, NormalForceLaw.SMOOTH_POSITIVE)
                        np.testing.assert_array_equal(config.eta_cop, np.tile([0., cop, 0., 0., 0., 0.], (2, 1)))
                        np.testing.assert_array_equal(config.eta_f, np.tile([force, 0., 0., 0., 0., 0.], (2, 1)))

    def test_normal_force_law_selection_and_legacy_args_both_scenes(self):
        for variant, module in [('flat', flat_cli), ('handheld', handheld_cli)]:
            parser = module.make_parser()
            scene = SimpleNamespace(variant=variant, contact_rest_gap=.248,
                                    geometry=SimpleNamespace(K_STIFF=module.K_STIFF))
            for law in NormalForceLaw:
                with self.subTest(variant=variant, law=law):
                    args = parser.parse_args(['--normal_force_law', law.value])
                    self.assertEqual(controller_config(scene, args).normal_force_law, law)
            args = parser.parse_args([])
            del args.normal_force_law
            self.assertEqual(controller_config(scene, args).normal_force_law,
                             NormalForceLaw.SMOOTH_POSITIVE)

    def test_handheld_rejects_invalid_normal_force_law(self):
        parser = handheld_cli.make_parser()
        for invalid in ('ideal', 'none', 'nan', ''):
            with self.subTest(invalid=invalid):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    parser.parse_args(['--normal_force_law', invalid])
                self.assertEqual(caught.exception.code, 2)

    def test_coefficients_reject_nonfinite_and_out_of_range_values(self):
        for module in (flat_cli, handheld_cli):
            parser = module.make_parser()
            for option in ('--eta_cop_y', '--eta_f_x', '--If1_x'):
                for invalid in ('nan', 'inf', '-inf', '-0.01', '1.01', 'invalid'):
                    with self.subTest(module=module.__name__, option=option, invalid=invalid):
                        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                            parser.parse_args([f'{option}={invalid}'])
                        self.assertEqual(caught.exception.code, 2)
                for boundary in ('0', '1'):
                    parsed = parser.parse_args([option, boundary])
                    self.assertEqual(getattr(parsed, 'eta_cop_y' if option == '--eta_cop_y' else 'If1_x'), float(boundary))


if __name__ == '__main__':
    unittest.main()
