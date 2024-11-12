class AntMazeRenderer(MazeRenderer):
    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        # Sjekk om denne faktisk lager et random goal.
        self.goal = self.env.set_target()
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]
        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)



