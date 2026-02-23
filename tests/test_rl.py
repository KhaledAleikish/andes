"""
Tests for the andes.rl Gymnasium environment.
"""

import unittest

import numpy as np

import andes
from andes.utils.paths import get_case

# Skip all tests if gymnasium is not installed
try:
    import gymnasium  # noqa: F401
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

CASE14 = get_case('ieee14/ieee14_esst3a.xlsx')


def _zero_reward(obs, action, env):
    return 0.0


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvSmoke(unittest.TestCase):
    """Basic smoke tests: create, reset, step."""

    @classmethod
    def setUpClass(cls):
        from andes.rl import AndesEnv
        cls.env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )

    def test_reset_returns_obs_info(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIn('t', info)
        self.assertAlmostEqual(info['t'], 0.0)

    def test_step_returns_5_tuple(self):
        self.env.reset(seed=42)
        action = np.zeros(self.env.action_space.shape)
        result = self.env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIn('t', info)

    def test_step_5_times(self):
        self.env.reset(seed=42)
        for _ in range(5):
            action = np.zeros(self.env.action_space.shape)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual(obs.shape, self.env.observation_space.shape)
            if terminated or truncated:
                break

    def test_ss_property(self):
        self.assertIsInstance(self.env.ss, andes.System)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvDeterminism(unittest.TestCase):
    """reinit-based reset should produce identical trajectories."""

    @classmethod
    def setUpClass(cls):
        from andes.rl import AndesEnv
        cls.env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )

    def _run_episode(self, n_steps=10):
        obs_list = []
        obs, _ = self.env.reset(seed=42)
        obs_list.append(obs.copy())
        for _ in range(n_steps):
            action = np.zeros(self.env.action_space.shape)
            obs, _, terminated, truncated, _ = self.env.step(action)
            obs_list.append(obs.copy())
            if terminated or truncated:
                break
        return np.array(obs_list)

    def test_two_episodes_identical(self):
        traj1 = self._run_episode()
        traj2 = self._run_episode()
        np.testing.assert_allclose(traj1, traj2, atol=1e-6)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvCustomReward(unittest.TestCase):
    """Custom reward_fn should be called."""

    def test_custom_reward_used(self):
        from andes.rl import AndesEnv

        called = []

        def my_reward(obs, action, env):
            called.append(1)
            return -42.0

        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=my_reward,
            dt=0.1,
            tf=2.0,
        )
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        _, reward, _, _, _ = env.step(action)
        self.assertEqual(reward, -42.0)
        self.assertTrue(len(called) > 0)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvDisturbance(unittest.TestCase):
    """disturbance_fn should be called on each reset."""

    def test_disturbance_called(self):
        from andes.rl import AndesEnv

        called = []

        def my_disturbance(env):
            called.append(1)

        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
            disturbance_fn=my_disturbance,
        )
        env.reset(seed=0)
        self.assertEqual(len(called), 1)
        env.reset(seed=1)
        self.assertEqual(len(called), 2)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvTruncation(unittest.TestCase):
    """Episode should truncate when t >= tf."""

    def test_truncated_at_tf(self):
        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.5,
            tf=2.0,
        )
        env.reset(seed=42)
        truncated = False
        for _ in range(100):  # should truncate well before 100 steps
            action = np.zeros(env.action_space.shape)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertGreaterEqual(info['t'], 2.0)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvMultipleObs(unittest.TestCase):
    """Multiple observation specs and idx selection."""

    def test_multiple_obs(self):
        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[
                ('GENROU', 'omega'),
                ('Bus', 'v'),
            ],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )
        obs, _ = env.reset(seed=42)
        n_genrou = env.ss.GENROU.n
        n_bus = env.ss.Bus.n
        self.assertEqual(obs.shape[0], n_genrou + n_bus)

    def test_obs_idx_selection(self):
        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[('Bus', 'v', [1, 5, 14])],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape[0], 3)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvModelAction(unittest.TestCase):
    """Model-level action spec (direct write)."""

    def test_model_action(self):
        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('TGOV1', 'paux0')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )
        n_tgov = env.ss.TGOV1.n
        self.assertEqual(env.action_space.shape[0], n_tgov)
        env.reset(seed=42)
        action = np.zeros(n_tgov)
        obs, _, _, _, _ = env.step(action)
        self.assertEqual(obs.shape, env.observation_space.shape)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvValidation(unittest.TestCase):
    """Init-time validation errors."""

    def test_invalid_target(self):
        from andes.rl import AndesEnv
        with self.assertRaises(ValueError):
            AndesEnv(
                case=CASE14,
                obs=[('GENROU', 'omega')],
                acts=[('NonExistentGroup', 'paux')],
                reward_fn=_zero_reward,
                dt=0.1,
                tf=2.0,
            )

    def test_invalid_setpoint(self):
        from andes.rl import AndesEnv
        with self.assertRaises(ValueError):
            AndesEnv(
                case=CASE14,
                obs=[('GENROU', 'omega')],
                acts=[('SynGen', 'nonexistent_setpoint')],
                reward_fn=_zero_reward,
                dt=0.1,
                tf=2.0,
            )

    def test_invalid_obs_model(self):
        from andes.rl import AndesEnv
        with self.assertRaises((ValueError, AttributeError)):
            AndesEnv(
                case=CASE14,
                obs=[('NONEXISTENT', 'omega')],
                acts=[('SynGen', 'paux')],
                reward_fn=_zero_reward,
                dt=0.1,
                tf=2.0,
            )

    def test_action_shape_mismatch(self):
        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )
        env.reset(seed=42)
        wrong_action = np.zeros(2)  # expected 5
        with self.assertRaises(ValueError):
            env.step(wrong_action)

    def test_missing_reward_fn(self):
        from andes.rl import AndesEnv
        with self.assertRaises(TypeError):
            AndesEnv(
                case=CASE14,
                obs=[('GENROU', 'omega')],
                acts=[('SynGen', 'paux')],
                reward_fn=None,
            )


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvTerminated(unittest.TestCase):
    """Simulation crash should set terminated=True."""

    def test_terminated_on_crash(self):
        from andes.rl import AndesEnv

        def sabotage(env):
            """Inject NaN into state variables to cause divergence."""
            env.ss.dae.x[:] = np.nan

        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=20.0,
            disturbance_fn=sabotage,
        )
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape)
        _, _, terminated, truncated, _ = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvTdsConfigValidation(unittest.TestCase):
    """Invalid tds_config keys should raise ValueError."""

    def test_invalid_tds_config_key(self):
        from andes.rl import AndesEnv
        with self.assertRaises(ValueError):
            AndesEnv(
                case=CASE14,
                obs=[('GENROU', 'omega')],
                acts=[('SynGen', 'paux')],
                reward_fn=_zero_reward,
                tds_config={'no_tqm': 1},  # typo for no_tqdm
            )


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed")
class TestAndesEnvGymnasiumChecker(unittest.TestCase):
    """Gymnasium's own env checker."""

    def test_check_env(self):
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            self.skipTest("gymnasium.utils.env_checker not available")

        from andes.rl import AndesEnv
        env = AndesEnv(
            case=CASE14,
            obs=[('GENROU', 'omega')],
            acts=[('SynGen', 'paux')],
            reward_fn=_zero_reward,
            dt=0.1,
            tf=2.0,
        )
        # check_env raises or warns on violations
        check_env(env)


if __name__ == '__main__':
    unittest.main()
