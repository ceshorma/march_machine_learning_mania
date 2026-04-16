import json
import tempfile
import unittest
from pathlib import Path

from automl_framework.architecture import build_architecture_report
from automl_framework.cli import default_registry_payload
from automl_framework.exploration import ExplorationPolicy
from automl_framework.generator import ExperimentGenerator, apply_change
from automl_framework.models import ConfigChange, ExperimentResult, Hypothesis, ProjectSpec
from automl_framework.registry import ExperimentRegistry
from examples.march_madness.adapter import MarchMadnessAdapter


class FrameworkCoreTests(unittest.TestCase):
    def test_apply_change_updates_nested_path(self):
        config = {'features': {'travel_distance': False}}
        changed = apply_change(config, ConfigChange(target='features.travel_distance', value=True))
        self.assertFalse(config['features']['travel_distance'])
        self.assertTrue(changed['features']['travel_distance'])

    def test_generator_picks_highest_priority_open_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = ExperimentRegistry(
                Path(tmp_dir) / 'registry.json',
                project=ProjectSpec(name='demo', objective='demo', metric='cv_brier'),
            )
            hypotheses = [
                Hypothesis(id='low', title='Low priority', description='low', change_kind='feature', priority=10, path_key='path:low'),
                Hypothesis(id='high', title='High priority', description='high', change_kind='feature', priority=99, path_key='path:high'),
            ]
            mutation_space = {
                'low': [ConfigChange(target='features.low', value=True)],
                'high': [ConfigChange(target='features.high', value=True)],
            }
            proposal = ExperimentGenerator().propose({'features': {}}, None, hypotheses, mutation_space, registry)
            self.assertEqual(proposal.hypothesis_id, 'high')
            self.assertTrue(proposal.config['features']['high'])
            self.assertEqual(proposal.config['id'], proposal.id)

    def test_registry_tracks_best_metric_and_attempts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = ExperimentRegistry(
                Path(tmp_dir) / 'registry.json',
                project=ProjectSpec(name='demo', objective='demo', metric='cv_brier'),
            )
            first = ExperimentResult(experiment_id='exp_001', status='completed', metrics={'cv_brier': 0.2}, path_key='feature:travel')
            first.improved = ExplorationPolicy().mark_improvement(first, registry)
            registry.record(first)
            second = ExperimentResult(experiment_id='exp_002', status='completed', metrics={'cv_brier': 0.19}, path_key='feature:travel')
            second.improved = ExplorationPolicy().mark_improvement(second, registry)
            registry.record(second)
            payload = json.loads(Path(tmp_dir, 'registry.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['summary']['best_experiment_id'], 'exp_002')
            self.assertEqual(payload['summary']['path_attempts']['feature:travel'], 2)
            self.assertAlmostEqual(payload['summary']['best_metric'], 0.19)

    def test_default_registry_payload_uses_project_metric_contract(self):
        project = ProjectSpec(name='demo', objective='demo', metric='roc_auc', metric_mode='max')
        payload = default_registry_payload(project)
        self.assertEqual(payload['summary']['metric'], 'roc_auc')
        self.assertEqual(payload['summary']['metric_mode'], 'max')

    def test_architecture_report_describes_core_and_adapter_split(self):
        repo_root = Path(__file__).resolve().parents[1]
        report = build_architecture_report(MarchMadnessAdapter(repo_root))
        self.assertIn('## Operational tree', report)
        self.assertIn('automl_framework.models', report)
        self.assertIn('### Core owns', report)
        self.assertIn('### Adapter owns', report)
        self.assertIn('examples/march_madness/adapter.py', report)


if __name__ == '__main__':
    unittest.main()
