MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
    # ,'antmaze-medium-diverse':(0, 9, 0, 12)
}

class AntRenderer(MazeRenderer):
    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = self.env.goal_location if hasattr(self.env, 'goal_location') else None
        self._background = None # Ukjent for nå
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